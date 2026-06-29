//
// RT64
//

#include "rt64_workload_queue.h"

#include "common/rt64_thread.h"

#include "rt64_present_queue.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <cstring>
#include <sys/system_properties.h>
#define RT64_WORKLOAD_ANDROID_LOG(...) ((void)0)

static bool rt64_workload_android_property_equals(const char *name, const char *expected) {
    char value[PROP_VALUE_MAX] = {};
    return __system_property_get(name, value) > 0 && std::strcmp(value, expected) == 0;
}

static bool rt64_workload_android_emulator_enabled() {
    return rt64_workload_android_property_equals("ro.kernel.qemu", "1") ||
        rt64_workload_android_property_equals("ro.boot.qemu", "1");
}
#else
#define RT64_WORKLOAD_ANDROID_LOG(...) ((void)0)

static bool rt64_workload_android_emulator_enabled() {
    return false;
}
#endif

#define ENABLE_HIGH_RESOLUTION_RENDERER 1

namespace RT64 {
    // WorkloadQueue

    WorkloadQueue::WorkloadQueue() {
        reset();
    }

    WorkloadQueue::~WorkloadQueue() {
        threadsRunning = false;
        cursorCondition.notify_all();
        idleCondition.notify_all();

        if (renderThread != nullptr) {
            renderThread->join();
            delete renderThread;
        }

        if (idleThread != nullptr) {
            idleThread->join();
            delete idleThread;
        }

        workloadIdCondition.notify_all();
    }

    void WorkloadQueue::reset() {
        for (Workload &w : workloads) {
            w.reset();
        }

        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = int(workloads.size()) - 1;
        workloadId = 0;
        lastPresentId = 0;
    }

    void WorkloadQueue::advanceToNextWorkload() {
        int nextWriteCursor = (writeCursor + 1) % workloads.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a workload being used by the GPU.
        bool waitForBarrier;
        do {
            const std::scoped_lock lock(cursorMutex);
            waitForBarrier = (nextWriteCursor == barrierCursor);
        } while (waitForBarrier);

        // Modify the cursor and notify anything waiting on the queue.
        {
            const std::scoped_lock lock(cursorMutex);
            writeCursor = nextWriteCursor;
        }

        cursorCondition.notify_all();
    }

    void WorkloadQueue::repeatLastWorkload() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t WorkloadQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(workloads.size()) - 1;
        }
    }

    void WorkloadQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void WorkloadQueue::waitForWorkloadId(uint64_t waitId) {
        std::unique_lock<std::mutex> workloadLock(workloadIdMutex);
        workloadIdCondition.wait(workloadLock, [&]() {
            return (waitId <= workloadId) || !threadsRunning;
        });
    }

    void WorkloadQueue::setup(const External &ext) {
        this->ext = ext;

        rspProcessor = std::make_unique<RSPProcessor>(ext.device);
        vertexProcessor = std::make_unique<VertexProcessor>(ext.device);
        framebufferRenderer = std::make_unique<FramebufferRenderer>(ext.workloadGraphicsWorker, true, ext.createdGraphicsAPI, ext.shaderLibrary);
        renderFramebufferManager = std::make_unique<RenderFramebufferManager>(ext.device);
        queryPool = ext.device->createQueryPool(2);

        projectionProcessor.setup(ext.workloadGraphicsWorker);
        transformProcessor.setup(ext.workloadGraphicsWorker);
        tileProcessor.setup(ext.workloadGraphicsWorker);
        lookAtProcessor.setup(ext.workloadGraphicsWorker);

        threadsRunning = true;
        renderThread = new std::thread(&WorkloadQueue::renderThreadLoop, this);
        idleThread = new std::thread(&WorkloadQueue::idleThreadLoop, this);
    }

    void WorkloadQueue::updateMultisampling() {
        renderFramebufferManager->destroyAll();
        dummyDepthTarget.reset();
        framebufferRenderer->updateMultisampling();
    }

    void WorkloadQueue::threadConfigurationUpdate(hlslpp::uint2 viFbSize, WorkloadConfiguration &workloadConfig) {
        const std::scoped_lock lock(ext.sharedResources->configurationMutex);
        const bool sizeChanged = ext.sharedResources->swapChainSizeChanged;
        ext.sharedResources->swapChainSizeChanged = false;
        
        // Retrieve the reference height to be used for determining the resolution scale. Impose a minimum in case
        // the game is using too small of a portion of the VI.
        const uint32_t MinimumReferenceHeight = 60;
        const uint32_t referenceHeight = (viFbSize[1] > 0) ? std::max(viFbSize[1], MinimumReferenceHeight) : 240;

        // Compute the aspect ratio to be used for the frame. Zelda's widescreen HUD patches assume
        // the N64's 4:3 presentation space even when the VI framebuffer height has a few extra rows.
        workloadConfig.aspectRatioSource = (4.0f / 3.0f);

        const auto ratioMode = ext.sharedResources->userConfig.aspectRatio;
        switch (ratioMode) {
        case UserConfiguration::AspectRatio::Expand:
            if ((ext.sharedResources->swapChainWidth > 0) && (ext.sharedResources->swapChainHeight > 0)) {
                const float derivedRatioTarget = float(ext.sharedResources->swapChainWidth) / float(ext.sharedResources->swapChainHeight);
                workloadConfig.aspectRatioTarget = std::max(derivedRatioTarget, workloadConfig.aspectRatioSource);
            }
            else {
                workloadConfig.aspectRatioTarget = workloadConfig.aspectRatioSource;
            }

            break;
        case UserConfiguration::AspectRatio::Manual:
            workloadConfig.aspectRatioTarget = float(ext.sharedResources->userConfig.aspectTarget);
            break;
        case UserConfiguration::AspectRatio::Original:
        default:
            workloadConfig.aspectRatioTarget = workloadConfig.aspectRatioSource;
            break;
        }

        // Compute the extended GBI aspect ratio percentage to be used for the frame.
        const auto extRatioMode = ext.sharedResources->userConfig.extAspectRatio;
        switch (extRatioMode) {
        case UserConfiguration::AspectRatio::Expand:
            workloadConfig.extAspectPercentage = 1.0f;
            break;
        case UserConfiguration::AspectRatio::Manual:
            if ((ext.sharedResources->swapChainWidth > 0) && (ext.sharedResources->swapChainHeight > 0)) {
                const float reducedExtTarget = float(ext.sharedResources->userConfig.extAspectTarget) - workloadConfig.aspectRatioSource;
                const float reducedDisplayTarget = workloadConfig.aspectRatioTarget - workloadConfig.aspectRatioSource;
                if ((reducedExtTarget > 0.0f) && (reducedDisplayTarget > 0.0f)) {
                    workloadConfig.extAspectPercentage = std::clamp((reducedExtTarget / reducedDisplayTarget), 0.0f, 1.0f);
                }
                else {
                    workloadConfig.extAspectPercentage = 0.0f;
                }
            }
            else {
                workloadConfig.extAspectPercentage = 0.0f;
            }

            break;
        case UserConfiguration::AspectRatio::Original:
        default:
            workloadConfig.extAspectPercentage = 0.0f;
            break;
        }

        // Compute the resolution scaling to be used for the frame.
        float resolutionMultiplier;
        const auto resolutionMode = ext.sharedResources->userConfig.resolution;
        switch (resolutionMode) {
        case UserConfiguration::Resolution::WindowIntegerScale:
            if (ext.sharedResources->swapChainHeight > 0) {
#           if defined(__ANDROID__)
                resolutionMultiplier = std::max(float(ext.sharedResources->swapChainHeight / referenceHeight), 1.0f);
#           else
                resolutionMultiplier = std::max(float((ext.sharedResources->swapChainHeight + referenceHeight - 1) / referenceHeight), 1.0f);
#           endif
            }
            else {
                resolutionMultiplier = 1.0f;
            }

            break;
        case UserConfiguration::Resolution::Manual:
            resolutionMultiplier = float(ext.sharedResources->userConfig.resolutionMultiplier);
            break;
        case UserConfiguration::Resolution::Original:
        default:
            resolutionMultiplier = 1.0f;
            break;
        }

        uint32_t msaaSampleCount = ext.sharedResources->userConfig.msaaSampleCount();

        // Build the resolution scale vector from the configuration.
        workloadConfig.aspectRatioScale = workloadConfig.aspectRatioTarget / workloadConfig.aspectRatioSource;
        workloadConfig.resolutionScale = { resolutionMultiplier * workloadConfig.aspectRatioScale, resolutionMultiplier };
        workloadConfig.downsampleMultiplier = ext.sharedResources->userConfig.downsampleMultiplier;
        ext.sharedResources->resolutionScale = workloadConfig.resolutionScale;

        // Find the target refresh rate from the configuration.
        const auto refreshRate = ext.sharedResources->userConfig.refreshRate;
        switch (refreshRate) {
        case UserConfiguration::RefreshRate::Display:
            workloadConfig.targetRate = ext.sharedResources->swapChainRate;
            break;
        case UserConfiguration::RefreshRate::Manual:
            workloadConfig.targetRate = ext.sharedResources->userConfig.refreshRateTarget;

            // Limit the target rate to the rate detected by the swap chain.
            if ((ext.sharedResources->swapChainRate > 0) && (workloadConfig.targetRate > ext.sharedResources->swapChainRate)) {
                workloadConfig.targetRate = ext.sharedResources->swapChainRate;
            }

            break;
        case UserConfiguration::RefreshRate::Original:
        default:
            workloadConfig.targetRate = 0;
            break;
        }

        // Store the rate that was chosen for the configuration.
        ext.sharedResources->targetRate = workloadConfig.targetRate;

#   if RT_ENABLED
        workloadConfig.raytracingEnabled = rtEnabled;

        if (workloadConfig.raytracingEnabled && (ext.sharedResources->rtConfigChanged || ext.sharedResources->fbConfigChanged || sizeChanged)) {
            // Only load the RT pipeline if the device supports it.
             if (ext.device->getCapabilities().raytracing && !ext.rtShaderCache->isSetup()) {
                 ext.rtShaderCache->setup();
            }

            framebufferRenderer->setRaytracingConfig(ext.sharedResources->rtConfig, ext.sharedResources->fbConfigChanged || sizeChanged);
            ext.sharedResources->rtConfigChanged = false;
        }
#   endif
        
        workloadConfig.postBlendNoise = ext.sharedResources->emulatorConfig.dither.postBlendNoise;
        workloadConfig.postBlendNoiseNegative = ext.sharedResources->emulatorConfig.dither.postBlendNoiseNegative;
        
        if (ext.sharedResources->fbConfigChanged || sizeChanged) {
            {
                // Wait until the other queue has stopped using the interpolated color targets.
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                InterpolatedFrameCounters &curFrameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                    return curFrameCounters.presented >= curFrameCounters.available;
                });
            }

            std::scoped_lock<std::mutex> managerLock(ext.sharedResources->managerMutex);
            FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
            RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
            renderFramebufferManager->destroyAll();
            targetManager.destroyAll();
            fbManager.destroyAllTileCopies();
            ext.sharedResources->fbConfigChanged = false;
            ext.sharedResources->interpolatedColorTargets.clear();
        }

        if (ext.sharedResources->userConfigChanged) {
            idleMutex.lock();
            idleActive = ext.sharedResources->userConfig.idleWorkActive;
            idleMutex.unlock();
            idleCondition.notify_all();
        }
    }

    void WorkloadQueue::threadConfigurationValidate() {
        const std::scoped_lock lock(ext.sharedResources->configurationMutex);
        if (ext.sharedResources->userConfigChanged) {
            ext.sharedResources->newConfigValidated = true;
            ext.sharedResources->userConfigChanged = false;
        }
    }
    
    void WorkloadQueue::threadRenderFrame(GameFrame &curFrame, const GameFrame &prevFrame, const WorkloadConfiguration &workloadConfig,
        const DebuggerRenderer &debuggerRenderer, const DebuggerCamera &debuggerCamera, float curFrameWeight, float prevFrameWeight,
        float deltaTimeMs, RenderTargetKey overrideTargetKey, int32_t overrideTargetFbPairIndex, RenderTarget *overrideTarget,
        uint32_t overrideTargetModifier, bool uploadVelocity, bool uploadExtras, bool interpolateTiles, bool interpolateLookAts)
    {
#   if ENABLE_HIGH_RESOLUTION_RENDERER
        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame begin curWorkloads=%zu prevMatched=%d targetRate=%u rt=%d",
            curFrame.workloads.size(), prevFrame.matched ? 1 : 0, workloadConfig.targetRate, workloadConfig.raytracingEnabled ? 1 : 0);
        std::scoped_lock<std::mutex> managerLock(ext.sharedResources->workloadMutex);
        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after workloadMutex lock");
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        const bool usingMSAA = (targetManager.multisampling.sampleCount > 1);

        rendererCPUProfiler.start();

        const bool aspectRatioAdjustment = (abs(workloadConfig.aspectRatioScale - 1.0f) > 1e-6f);
        const bool processProjections = aspectRatioAdjustment || prevFrame.matched|| curFrame.isDebuggerCameraEnabled(*this);
        bool uploadProjections = false;
        if (processProjections) {
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before projection process");
            ProjectionProcessor::ProcessParams projParams;
            projParams.worker = ext.workloadGraphicsWorker;
            projParams.workloadQueue = this;
            projParams.curFrame = &curFrame;
            projParams.prevFrame = &prevFrame;
            projParams.curFrameWeight = curFrameWeight;
            projParams.prevFrameWeight = prevFrameWeight;
            projParams.aspectRatioScale = workloadConfig.aspectRatioScale;
            projectionProcessor.process(projParams);
            projectionProcessor.upload(projParams);
            uploadProjections = true;
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after projection process");
        }

        const bool processTransforms = prevFrame.matched;
        bool uploadTransforms = false;
        if (processTransforms) {
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before transform process");
            TransformProcessor::ProcessParams transformParams;
            transformParams.worker = ext.workloadGraphicsWorker;
            transformParams.workloadQueue = this;
            transformParams.curFrame = &curFrame;
            transformParams.prevFrame = &prevFrame;
            transformParams.curFrameWeight = curFrameWeight;
            transformParams.prevFrameWeight = prevFrameWeight;
            transformProcessor.process(transformParams);
            transformProcessor.upload(transformParams);
            uploadTransforms = true;
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after transform process");
        }

        bool uploadTiles = false;
        if (interpolateTiles) {
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before tile process");
            TileProcessor::ProcessParams tileParams;
            tileParams.worker = ext.workloadGraphicsWorker;
            tileParams.workloadQueue = this;
            tileParams.curFrame = &curFrame;
            tileParams.prevFrame = &prevFrame;
            tileParams.curFrameWeight = curFrameWeight;
            tileParams.prevFrameWeight = prevFrameWeight;
            tileProcessor.process(tileParams);
            tileProcessor.upload(tileParams);
            uploadTiles = true;
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after tile process");
        }

        bool uploadLookAts = false;
        if (interpolateLookAts) {
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before lookAt process");
            LookAtProcessor::ProcessParams lookAtParams;
            lookAtParams.worker = ext.workloadGraphicsWorker;
            lookAtParams.workloadQueue = this;
            lookAtParams.curFrame = &curFrame;
            lookAtParams.prevFrame = &prevFrame;
            lookAtParams.curFrameWeight = curFrameWeight;
            lookAtParams.prevFrameWeight = prevFrameWeight;
            lookAtProcessor.process(lookAtParams);
            lookAtProcessor.upload(lookAtParams);
            uploadLookAts = true;
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after lookAt process");
        }

        // Reset the max height tracking for all active framebuffers.
        fbManager.resetTracking();

        if ((overrideTarget != nullptr) && !usingMSAA) {
            targetManager.setOverride(overrideTargetKey, overrideTarget);
        }

        for (uint32_t w = 0; w < curFrame.workloads.size(); w++) {
            Workload &workload = workloads[curFrame.workloads[w]];
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame workload begin index=%u cursor=%u fbPairs=%u gameCalls=%u gpuTiles=%zu",
                w, curFrame.workloads[w], workload.fbPairCount, workload.gameCallCount, workload.drawData.gpuTiles.size());

            // There's no guarantee the RSP was processed if framebuffers were not rendered.
            const bool processRSP = true;
            if (processRSP) {
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before RSP process");
                workload.resetRSPOutputBuffers();

                RSPProcessor::ProcessParams rspParams;
                rspParams.worker = ext.workloadGraphicsWorker;
                rspParams.drawData = &workload.drawData;
                rspParams.drawBuffers = &workload.drawBuffers;
                rspParams.outputBuffers = &workload.outputBuffers;
                rspParams.prevFrameWeight = prevFrameWeight;
                rspParams.curFrameWeight = curFrameWeight;
                rspProcessor->process(rspParams);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after RSP process");
            }

            const bool processWorldVertices = prevFrame.matched;
            if (processWorldVertices) {
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before vertex process");
                workload.resetWorldOutputBuffers();

                VertexProcessor::ProcessParams vertexParams;
                vertexParams.worker = ext.workloadGraphicsWorker;
                vertexParams.drawData = &workload.drawData;
                vertexParams.drawBuffers = &workload.drawBuffers;
                vertexParams.outputBuffers = &workload.outputBuffers;
                vertexParams.curFrameWeight = curFrameWeight;
                vertexParams.prevFrameWeight = prevFrameWeight;
                vertexProcessor->process(vertexParams);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after vertex process");
            }

            hlslpp::float2 fixedResScale;
            Framebuffer *colorFb;
            Framebuffer *depthFb;
            uint32_t nativeColorWidth;
            uint32_t nativeColorHeight;
            uint32_t targetWidth;
            uint32_t targetHeight;
            uint32_t targetMisalignX;
            uint32_t rtWidth;
            uint32_t rtHeight;
            RenderTarget *colorTarget;
            RenderTarget *depthTarget;
            RenderFramebufferKey fbKey;
            auto getTargetsFromPair = [&](uint32_t f) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                const auto &colorImg = fbPair.colorImage;
                const auto &depthImg = fbPair.depthImage;
                fixedResScale = workloadConfig.resolutionScale;
                if (!fbPair.drawColorRect.isEmpty()) {
                    colorFb = nullptr;
                    depthFb = nullptr;
                    nativeColorWidth = colorImg.width;
                    nativeColorHeight = fbPair.drawColorRect.bottom(true);

                    // When the target is much bigger than the reference height, we reduce the resolution scaling (but clamped to 1.0).
                    const uint32_t heightThreshold = (workload.viFbSize[1] > 0) ? ((workload.viFbSize[1] * 3) / 2) : 360;
                    uint32_t downsampleMultiplier = workloadConfig.downsampleMultiplier;
                    if ((nativeColorHeight >= heightThreshold) && (fixedResScale[1] >= 2.0f)) {
                        fixedResScale = hlslpp::max(fixedResScale / 2.0f, hlslpp::float2(1.0f, 1.0f));
                        downsampleMultiplier = std::max(downsampleMultiplier / 2U, 1U);
                    }

                    if (fbPair.depthRead || fbPair.depthWrite || fbPair.fastPaths.clearDepthOnly) {
                        uint32_t depthAddress = fbPair.fastPaths.clearDepthOnly ? colorImg.address : depthImg.address;
                        depthFb = &fbManager.get(depthAddress, G_IM_SIZ_16b, nativeColorWidth, nativeColorHeight);
                        depthFb->everUsedAsDepth = true;
                    }
                    else {
                        depthFb = nullptr;
                    }

                    // Ensure dimensions are the same for the color and depth targets based on their previous sizes.
                    fbKey = RenderFramebufferKey();

                    if (!fbPair.fastPaths.clearDepthOnly) {
                        colorFb = &fbManager.get(colorImg.address, colorImg.siz, nativeColorWidth, nativeColorHeight);
                    }

                    if (colorFb != nullptr) {
                        fbKey.colorTargetKey = RenderTargetKey(colorFb->addressStart, colorFb->width, colorFb->siz, Framebuffer::Type::Color);
                        colorTarget = &targetManager.get(fbKey.colorTargetKey);
                    }
                    else {
                        colorTarget = nullptr;
                    }

                    // Apply the modifier key if we retrieved the override target.
                    if ((colorTarget != nullptr) && (colorTarget == overrideTarget)) {
                        fbKey.modifierKey = overrideTargetModifier;
                    }

                    fixedResScale = RenderTarget::computeFixedResolutionScale(colorImg.width, fixedResScale);
                    RenderTarget::computeScaledSize(nativeColorWidth, nativeColorHeight, fixedResScale, targetWidth, targetHeight, targetMisalignX);

                    rtWidth = targetWidth;
                    rtHeight = targetHeight;

                    // The desired size should not be less than the existing size of the color and depth targets.
                    RenderTarget *chosenRt = nullptr;
                    if (depthFb != nullptr) {
                        fbKey.depthTargetKey = RenderTargetKey(depthFb->addressStart, depthFb->width, depthFb->siz, Framebuffer::Type::Depth);
                        depthTarget = &targetManager.get(fbKey.depthTargetKey);
                        depthTarget->resolutionScale = fixedResScale;
                        rtWidth = std::max(rtWidth, depthTarget->width);
                        rtHeight = std::max(rtHeight, depthTarget->height);
                        chosenRt = depthTarget;
                    }
                    else {
                        depthTarget = nullptr;
                    }

                    if (colorTarget != nullptr) {
                        rtWidth = std::max(rtWidth, colorTarget->width);
                        rtHeight = std::max(rtHeight, colorTarget->height);
                        chosenRt = colorTarget;
                    }

                    assert(chosenRt != nullptr);
                    chosenRt->resolutionScale = fixedResScale;
                    chosenRt->downsampleMultiplier = downsampleMultiplier;
                    chosenRt->misalignX = targetMisalignX;
                    chosenRt->invMisalignX = (targetMisalignX > 0) ? (std::lround(fixedResScale.y) - targetMisalignX) : 0;

                    assert((colorTarget != nullptr) || (depthTarget != nullptr));
                    return true;
                }
                else {
                    return false;
                }
            };

            thread_local std::unordered_set<RenderTarget *> resizedTargets;
            thread_local std::vector<std::pair<RenderTarget *, RenderTarget *>> colorDepthPairs;
            resizedTargets.clear();
            colorDepthPairs.clear();

            const uint32_t fbPairCount = (debuggerRenderer.framebufferIndex >= 0) ? (debuggerRenderer.framebufferIndex + 1) : workload.fbPairCount;
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before resize/setup fbPairCount=%u", fbPairCount);
            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
#           if RT_ENABLED
                for (uint32_t p = 0; p < fbPair.projectionCount; p++) {
                    const Projection &proj = fbPair.projections[p];
                    const bool perspProj = (proj.type == Projection::Type::Perspective);
                    const bool rtProj = (perspProj && workloadConfig.raytracingEnabled && fbPair.depthWrite); // TODO: Move this condition out of here, ideally by moving the shader submission elsewhere.
                    if (!rtProj) {
                        continue;
                    }

                    // Submit RT shaders if it's an RT proj.
                    for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                        const GameCall &call = proj.gameCalls[d];
                        ext.rtShaderCache->submit(call.shaderDesc);
                    }
                }
#           endif

                // Resize the render targets for this framebuffer pair if necessary.
                if (getTargetsFromPair(f)) {
                    // Resize the native target buffers.
                    if (colorFb != nullptr) {
                        colorFb->nativeTarget.resetBufferHistory();
                    }

                    if (depthFb != nullptr) {
                        depthFb->nativeTarget.resetBufferHistory();
                    }

                    if ((colorTarget != nullptr) && colorTarget->resize(ext.workloadGraphicsWorker, rtWidth, rtHeight)) {
                        resizedTargets.emplace(colorTarget);
                        colorFb->readHeight = 0;
                    }

                    // Set up the dummy target used for rendering the depth if no depth framebuffer is active.
                    if (depthFb == nullptr) {
                        if (dummyDepthTarget == nullptr) {
                            dummyDepthTarget = std::make_unique<RenderTarget>(0, Framebuffer::Type::Depth, targetManager.multisampling, targetManager.usesHDR);
                            dummyDepthTarget->setupDepth(ext.workloadGraphicsWorker, rtWidth, rtHeight);
                        }

                        if ((dummyDepthTarget != nullptr) && dummyDepthTarget->resize(ext.workloadGraphicsWorker, rtWidth, rtHeight)) {
                            resizedTargets.emplace(dummyDepthTarget.get());
                        }

                        if (colorTarget != nullptr) {
                            colorDepthPairs.emplace_back(colorTarget, dummyDepthTarget.get());
                        }
                    }
                    else if (depthTarget != nullptr) {
                        if (colorTarget != nullptr) {
                            colorDepthPairs.emplace_back(colorTarget, depthTarget);
                        }

                        if (depthTarget->resize(ext.workloadGraphicsWorker, rtWidth, rtHeight)) {
                            resizedTargets.emplace(depthTarget);
                            depthFb->readHeight = 0;
                        }
                    }
                }

                fbManager.setupOperations(ext.workloadGraphicsWorker, fbPair.startFbOperations, fixedResScale, targetManager, &resizedTargets);
                fbManager.setupOperations(ext.workloadGraphicsWorker, fbPair.endFbOperations, fixedResScale, targetManager, &resizedTargets);
            }
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after resize/setup resizedTargets=%zu", resizedTargets.size());

            // Make sure all depth targets are at least bigger than their corresponding color targets.
            for (auto colorDepthPair : colorDepthPairs) {
                if (colorDepthPair.second->resize(ext.workloadGraphicsWorker, colorDepthPair.first->width, colorDepthPair.first->height)) {
                    resizedTargets.emplace(colorDepthPair.second);
                }
            }

            for (RenderTarget *renderTarget : resizedTargets) {
                renderFramebufferManager->destroyAllWithRenderTarget(renderTarget);
            }

            uint32_t gameCallCursor = 0;
            const uint32_t gameCallCountMax = (debuggerRenderer.globalDrawCallIndex >= 0) ? (debuggerRenderer.globalDrawCallIndex + 1) : workload.gameCallCount;
            thread_local std::vector<BufferUploader *> bufferUploaders;
            bufferUploaders.clear();

            // Indicate to the texture cache the textures must not be deleted.
            ext.textureCache->incrementLock();

            // Reset the texture cache vectors for the framebuffer renderer.
            framebufferRenderer->updateTextureCache(ext.textureCache);
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after texture cache update");

            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                fbManager.performDiscards(fbPair.startFbDiscards);
            }
            
            // Add all framebuffer pairs to the framebuffer renderer and setup the operations.
            scratchFbChangePool.reset();
            fbManager.resetOperations();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before resetFramebuffers");
            framebufferRenderer->resetFramebuffers(ext.workloadGraphicsWorker, ubershadersVisible, workload.extended.ditherNoiseStrength, targetManager.multisampling);
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after resetFramebuffers");

#       if RT_ENABLED
            if (workloadConfig.raytracingEnabled) {
                framebufferRenderer->resetRaytracing(ext.rtShaderCache, ext.blueNoiseTexture);
            }
#       endif

            for (uint32_t f = 0; f < fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                if (getTargetsFromPair(f)) {
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before addFramebuffer f=%u target=%ux%u native=%ux%u",
                        f, targetWidth, targetHeight, nativeColorWidth, nativeColorHeight);
                    RenderFramebufferStorage &fbStorage = renderFramebufferManager->get(fbKey, colorTarget, (depthTarget != nullptr) ? depthTarget : dummyDepthTarget.get());
                    FramebufferRenderer::DrawParams drawParams;
                    drawParams.worker = ext.workloadGraphicsWorker;
                    drawParams.fbStorage = &fbStorage;
                    drawParams.curWorkload = &workload;
                    drawParams.fbPairIndex = f;
                    drawParams.fbWidth = nativeColorWidth;
                    drawParams.fbHeight = nativeColorHeight;
                    drawParams.targetWidth = targetWidth;
                    drawParams.targetHeight = targetHeight;
                    drawParams.rasterShaderCache = ext.rasterShaderCache;
                    drawParams.resolutionScale = fixedResScale;
                    drawParams.aspectRatioSource = workloadConfig.aspectRatioSource;
                    drawParams.aspectRatioTarget = workloadConfig.aspectRatioTarget;
                    drawParams.extAspectPercentage = workloadConfig.extAspectPercentage;
                    const float horizontalMisalignment = (colorTarget != nullptr) ? float(colorTarget->misalignX) : float(depthTarget->misalignX);
#if defined(__ANDROID__)
                    drawParams.horizontalMisalignment = (workloadConfig.aspectRatioScale != 1.0f) ? 0.0f : horizontalMisalignment;
#else
                    drawParams.horizontalMisalignment = horizontalMisalignment;
#endif
                    drawParams.presetScene = curFrame.presetScene;
                    drawParams.rtEnabled = workloadConfig.raytracingEnabled;
                    drawParams.submissionFrame = workload.submissionFrame;
                    drawParams.deltaTimeMs = deltaTimeMs;
                    drawParams.ubershadersOnly = ubershadersOnly;
                    drawParams.postBlendNoise = workloadConfig.postBlendNoise;
                    drawParams.postBlendNoiseNegative = workloadConfig.postBlendNoiseNegative;
                    drawParams.maxGameCall = std::min(gameCallCountMax - gameCallCursor, fbPair.gameCallCount);
                    framebufferRenderer->addFramebuffer(drawParams);
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after addFramebuffer f=%u", f);
                }
                
                gameCallCursor += fbPair.gameCallCount;
            }

            // Create all GPU tile mappings and upload them.
            if (!workload.drawData.gpuTiles.empty()) {
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before createGPUTiles count=%zu", workload.drawData.gpuTiles.size());
                std::pair<size_t, size_t> gpuTileRange;
                gpuTileRange.first = 0;
                gpuTileRange.second = workload.drawData.gpuTiles.size();
                framebufferRenderer->createGPUTiles(workload.drawData.callTiles.data(), uint32_t(workload.drawData.gpuTiles.size()),
                    workload.drawData.gpuTiles.data(), &fbManager, ext.textureCache, workload.submissionFrame);

                // Upload the GPU tiles.
                ext.workloadTilesUploader->submit(ext.workloadGraphicsWorker, {
                    { workload.drawData.gpuTiles.data(), gpuTileRange, sizeof(interop::GPUTile), RenderBufferFlag::STORAGE, { }, &workload.drawBuffers.gpuTilesBuffer}
                });

                bufferUploaders.emplace_back(ext.workloadTilesUploader);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after createGPUTiles");
            }

            if (uploadVelocity) {
                bufferUploaders.emplace_back(ext.workloadVelocityUploader);
                uploadVelocity = false;
            }

            if (uploadExtras) {
                bufferUploaders.emplace_back(ext.workloadExtrasUploader);
                uploadExtras = false;
            }

            if (uploadProjections) {
                bufferUploaders.emplace_back(projectionProcessor.bufferUploader.get());
                uploadProjections = false;
            }

            if (uploadTransforms) {
                bufferUploaders.emplace_back(transformProcessor.bufferUploader.get());
                uploadTransforms = false;
            }

            if (uploadTiles) {
                bufferUploaders.emplace_back(tileProcessor.bufferUploader.get());
                uploadTiles = false;
            }

            if (uploadLookAts) {
                bufferUploaders.emplace_back(lookAtProcessor.bufferUploader.get());
                uploadLookAts = false;
            }

#       if RT_ENABLED
            if (workloadConfig.raytracingEnabled) {
                ext.rtShaderCache->setNextState();
            }
#       endif

            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before workerMutex lock uploaders=%zu", bufferUploaders.size());
            workerMutex.lock();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after workerMutex lock");
            ext.workloadGraphicsWorker->commandList->begin();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after commandList begin");
            ext.workloadGraphicsWorker->commandList->resetQueryPool(queryPool.get(), 0, 2);
            ext.workloadGraphicsWorker->commandList->writeTimestamp(queryPool.get(), 0);
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before endFramebuffers");
            framebufferRenderer->endFramebuffers(ext.workloadGraphicsWorker, &workload.drawBuffers, &workload.outputBuffers, workloadConfig.raytracingEnabled);
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after endFramebuffers");
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before recordSetup");
            framebufferRenderer->recordSetup(ext.workloadGraphicsWorker, bufferUploaders, processRSP ? rspProcessor.get() : nullptr, processWorldVertices ? vertexProcessor.get() : nullptr, &workload.outputBuffers, workloadConfig.raytracingEnabled);
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after recordSetup");
            
            // Record all framebuffer pairs.
            uint32_t framebufferIndex = 0;
            for (uint32_t f = 0; f < fbPairCount; f++) {
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before record fbPair f=%u", f);
                const FramebufferPair &fbPair = workload.fbPairs[f];
                bool validTargets = getTargetsFromPair(f);
                fbManager.recordOperations(ext.workloadGraphicsWorker, &workload.fbChangePool, &workload.fbStorage, ext.shaderLibrary, ext.textureCache,
                    fbPair.startFbOperations, targetManager, fixedResScale, f, workload.submissionFrame);

                if (validTargets) {
                    const auto &colorImg = fbPair.colorImage;
                    const auto &depthImg = fbPair.depthImage;
                    bool colorFormatUpdated = false;
                    if (colorFb != nullptr) {
                        if (colorImg.formatChanged) {
                            colorFb->discardLastWrite();
                        }
                        else if (colorFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                            RenderTargetKey otherColorTargetKey(colorFb->addressStart, colorFb->width, colorFb->siz, colorFb->lastWriteType);
                            RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey);
                            if (!otherColorTarget.isEmpty()) {
                                const FixedRect &r = colorFb->lastWriteRect;
                                colorTarget->copyFromTarget(ext.workloadGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                                colorFb->discardLastWrite();
                                colorFormatUpdated = true;
                            }
                        }

                        if (colorImg.formatChanged) {
                            colorTarget->clearColorTarget(ext.workloadGraphicsWorker);
                            colorFb->readHeight = 0;
                        }

                        if (colorFb->height > colorFb->readHeight) {
                            uint32_t readRowCount = colorFb->height - colorFb->readHeight;
                            FramebufferChange *colorFbChange = colorFb->readChangeFromStorage(ext.workloadGraphicsWorker, workload.fbStorage, scratchFbChangePool,
                                Framebuffer::Type::Color, colorImg.fmt, f, colorFb->readHeight, readRowCount, ext.shaderLibrary);

                            if (colorFbChange != nullptr) {
                                colorTarget->copyFromChanges(ext.workloadGraphicsWorker, *colorFbChange, colorFb->width, readRowCount, colorFb->readHeight, ext.shaderLibrary);
                            }

                            colorFb->readHeight = colorFb->height;
                        }
                    }

                    bool depthFormatUpdated = false;
                    bool depthFbChanged = false;
                    bool depthFbTypeChanged = false;
                    if (depthFb != nullptr) {
                        depthFbTypeChanged = (depthFb->lastWriteType == Framebuffer::Type::Color);

                        bool imgFormatChanged = (colorFb == nullptr) ? colorImg.formatChanged : depthImg.formatChanged;
                        if (imgFormatChanged) {
                            depthFb->discardLastWrite();
                        }
                        else if (depthFb->isLastWriteDifferent(Framebuffer::Type::Depth)) {
                            RenderTargetKey otherDepthTargetKey(depthFb->addressStart, depthFb->width, depthFb->siz, depthFb->lastWriteType);
                            RenderTarget &otherDepthTarget = targetManager.get(otherDepthTargetKey);
                            if (!otherDepthTarget.isEmpty()) {
                                const FixedRect &r = depthFb->lastWriteRect;
                                depthTarget->copyFromTarget(ext.workloadGraphicsWorker, &otherDepthTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                                depthFb->discardLastWrite();
                                depthFormatUpdated = true;
                            }
                        }

                        if (imgFormatChanged) {
                            depthTarget->clearDepthTarget(ext.workloadGraphicsWorker);
                            depthFb->readHeight = 0;
                        }

                        if (depthFb->height > depthFb->readHeight) {
                            uint32_t readRowCount = depthFb->height - depthFb->readHeight;
                            FramebufferChange *depthFbChange = depthFb->readChangeFromStorage(ext.workloadGraphicsWorker, workload.fbStorage, scratchFbChangePool, Framebuffer::Type::Depth,
                                G_IM_FMT_DEPTH, f, depthFb->readHeight, readRowCount, ext.shaderLibrary);

                            if (depthFbChange != nullptr) {
                                depthTarget->copyFromChanges(ext.workloadGraphicsWorker, *depthFbChange, depthFb->width, readRowCount, depthFb->readHeight, ext.shaderLibrary);
                                depthFbChanged = true;
                            }

                            depthFb->readHeight = depthFb->height;
                        }
                    }
                    
                    framebufferRenderer->recordFramebuffer(ext.workloadGraphicsWorker, framebufferIndex++);

                    // Transition the render targets in case the present queue will show them so it doesn't have to perform transitions.
                    if (colorTarget != nullptr && depthTarget != nullptr) {
                        RenderTextureBarrier textureBarriers[] = {
                            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::SHADER_READ),
                            RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::SHADER_READ)
                        };

                        ext.workloadGraphicsWorker->commandList->barriers(RenderBarrierStage::GRAPHICS, textureBarriers, uint32_t(std::size(textureBarriers)));
                    }
                    else {
                        RenderTarget *chosenTarget = (colorTarget != nullptr) ? colorTarget : depthTarget;
                        ext.workloadGraphicsWorker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(chosenTarget->texture.get(), RenderTextureLayout::SHADER_READ));
                    }

                    // Do the resolve if using MSAA while target override is active and we're on the correct framebuffer pair index.
                    if (usingMSAA && (overrideTarget != nullptr) && ((uint32_t)overrideTargetFbPairIndex == f)) {
                        overrideTarget->resize(ext.workloadGraphicsWorker, colorTarget->width, colorTarget->height);
                        overrideTarget->resolveFromTarget(ext.workloadGraphicsWorker, colorTarget, ext.shaderLibrary);
                    }

                    const uint64_t writeTimestamp = fbManager.nextWriteTimestamp();
                    FixedRect depthFbRect;
                    if (colorFb != nullptr) {
                        colorFb->lastWriteRect.merge(fbPair.drawColorRect.scaled(fixedResScale.x, fixedResScale.y));
                        colorFb->lastWriteType = Framebuffer::Type::Color;
                        colorFb->lastWriteFmt = colorImg.fmt;
                        colorFb->lastWriteTimestamp = writeTimestamp;
                        depthFbRect = fbPair.drawDepthRect;
                    }
                    else {
                        depthFbRect = fbPair.drawColorRect;
                    }
                    
                    const bool depthWrite = ((colorFb == nullptr) || depthFbChanged || depthFbTypeChanged || fbPair.depthWrite) && (depthFb != nullptr);
                    if (depthWrite && !depthFbRect.isNull()) {
                        depthFb->lastWriteRect.merge(depthFbRect.scaled(fixedResScale.x, fixedResScale.y));
                        depthFb->lastWriteType = Framebuffer::Type::Depth;
                        depthFb->lastWriteFmt = G_IM_FMT_DEPTH;
                        depthFb->lastWriteTimestamp = writeTimestamp;
                    }
                }
                
                fbManager.recordOperations(ext.workloadGraphicsWorker, &workload.fbChangePool, &workload.fbStorage, ext.shaderLibrary, ext.textureCache,
                    fbPair.endFbOperations, targetManager, fixedResScale, f, workload.submissionFrame);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after record fbPair f=%u", f);
            }

            ext.workloadGraphicsWorker->commandList->writeTimestamp(queryPool.get(), 1);
            ext.workloadGraphicsWorker->commandList->end();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after commandList end");
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before waitForUploaders");
            framebufferRenderer->waitForUploaders();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after waitForUploaders");
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before worker execute");
            ext.workloadGraphicsWorker->execute();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after worker execute");
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame before worker wait");
            ext.workloadGraphicsWorker->wait();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after worker wait");
            workerMutex.unlock();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame after workerMutex unlock");

            // Update the GPU profiler with the results from the timestamps of the frame.
            queryPool->queryResults();
            const uint64_t *frameTimestamps = queryPool->getResults();
            rendererGPUProfiler.log(double(frameTimestamps[1] - frameTimestamps[0]) / 1000000.0);

            // Indicate to the texture cache it's safe to delete the textures if no locks are active.
            ext.textureCache->decrementLock();
            RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame workload end index=%u", w);
        }

        if ((overrideTarget != nullptr) && !usingMSAA) {
            targetManager.removeOverride(overrideTargetKey);
        }

        framebufferRenderer->advanceFrame(workloadConfig.raytracingEnabled);
        rendererCPUProfiler.end();
        rendererCPUProfiler.log();
        rendererCPUProfiler.reset();
        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadRenderFrame end");
#   endif
    }

    void WorkloadQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % workloads.size();
    }

    void WorkloadQueue::threadAdvanceWorkloadId(uint64_t newWorkloadId) {
        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadAdvanceWorkloadId begin id=%llu", static_cast<unsigned long long>(newWorkloadId));
        {
            std::scoped_lock<std::mutex> cursorLock(workloadIdMutex);
            workloadId = newWorkloadId;
        }

        workloadIdCondition.notify_all();
        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::threadAdvanceWorkloadId end id=%llu", static_cast<unsigned long long>(newWorkloadId));
    }

    void WorkloadQueue::renderThreadLoop() {
        Thread::setCurrentThreadName("RT64 Workload");

        WorkloadConfiguration workloadConfig;
        int64_t logicalTicks = 0;
        int64_t displayTicks = 0;
        uint32_t originalRateForTicks = 0;
        uint32_t displayRateForTicks = 0;
        int processCursor = -1;
        bool frameReduction = false;
        while (threadsRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                cursorCondition.wait(cursorLock, [&]() {
                    return (writeCursor != threadCursor) || !threadsRunning;
                });

                if (threadsRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % workloads.size();
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop got cursor=%d writeCursor=%u nextThreadCursor=%u",
                        processCursor, writeCursor, threadCursor);
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                Workload &workload = workloads[processCursor];
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before waitForPresentId cursor=%d presentId=%llu workloadId=%llu",
                    processCursor, static_cast<unsigned long long>(workload.presentId), static_cast<unsigned long long>(workload.workloadId));
                ext.presentQueue->waitForPresentId(workload.presentId);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after waitForPresentId cursor=%d", processCursor);

                if (!threadsRunning) {
                    continue;
                }

                ElapsedTimer workloadTimer;
                workloadProfiler.start();
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before configuration update");
                threadConfigurationUpdate(workload.viFbSize, workloadConfig);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after configuration update vi=%ux%u targetRate=%u",
                    workload.viFbSize[0], workload.viFbSize[1], workloadConfig.targetRate);

                // FIXME: This is a very hacky way to find out if we need to advance the frame if the workload was paused for the first time.
                if (!workload.paused || (!gameFrames[curFrameIndex].workloads.empty() && (gameFrames[curFrameIndex].workloads[0] != (uint32_t)processCursor))) {
                    prevFrameIndex = curFrameIndex;
                    curFrameIndex = (curFrameIndex + 1) % gameFrames.size();
                }
                
                // TODO: The frame detection needs to be more elaborate than just matching one workload to one frame.
                GameFrame &curFrame = gameFrames[curFrameIndex];
                const GameFrame &prevFrame = gameFrames[prevFrameIndex];
                uint32_t workloadIndex = processCursor;
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before curFrame.set");
                curFrame.set(*this, &workloadIndex, 1);
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after curFrame.set curIndex=%u prevIndex=%u matched=%d",
                    curFrameIndex, prevFrameIndex, curFrame.matched ? 1 : 0);

                // Detect the color image to interpolate for this workload.
                RenderTargetKey interpolationTargetKey;
                int32_t interpolationTargetFbPairIndex = -1;
                {
                    std::scoped_lock<std::mutex> managerLock(ext.sharedResources->managerMutex);
                    FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
                    std::vector<uint32_t> &colorVector = ext.sharedResources->colorImageAddressVector;
                    std::unordered_set<uint32_t> &colorSet = ext.sharedResources->colorImageAddressSet;
                    colorVector.clear();
                    colorSet.clear();
                    for (int32_t f = workload.fbPairCount - 1; f >= 0; f--) {
                        const FramebufferPair &fbPair = workload.fbPairs[f];
                        bool interpolationCandidate = fbPair.earlyPresentCandidate();
                        if (fbPair.drawColorRect.isEmpty()) {
                            continue;
                        }

                        const auto &colorImg = fbPair.colorImage;
                        if (colorSet.find(colorImg.address) != colorSet.end()) {
                            continue;
                        }
                        else {
                            if (interpolationCandidate) {
                                colorVector.push_back(colorImg.address);
                            }

                            colorSet.insert(colorImg.address);
                        }

                        if (!interpolationCandidate || !interpolationTargetKey.isEmpty()) {
                            continue;
                        }

                        Framebuffer *interpolationFb = fbManager.find(colorImg.address);
                        if ((interpolationFb != nullptr) && interpolationFb->interpolationEnabled) {
                            interpolationTargetKey.fbType = Framebuffer::Type::Color;
                            interpolationTargetKey.address = fbPair.colorImage.address;
                            interpolationTargetKey.siz = fbPair.colorImage.siz;
                            interpolationTargetKey.width = fbPair.colorImage.width;
                            interpolationTargetFbPairIndex = f;
                        }
                    }
                }

                float prevFrameWeight = 0.0f;
                float curFrameWeight = 1.0f;
                float deltaTimeMs = 1.0f / 30.0f;
                const bool requiresFrameMatching = (workloadConfig.targetRate > 0) || workloadConfig.raytracingEnabled;
                bool generateInterpolatedFrames = false;
                bool velocityUploaderUsed = false;
                bool tileInterpolationUsed = false;
                bool lookAtInterpolationUsed = false;
                if (requiresFrameMatching) {
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before frame match");
                    matchingProfiler.reset();
                    matchingProfiler.start();
                    curFrame.match(ext.workloadGraphicsWorker, *this, prevFrame, ext.workloadVelocityUploader, velocityUploaderUsed, tileInterpolationUsed, lookAtInterpolationUsed);
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after frame match velocity=%d tile=%d lookAt=%d",
                        velocityUploaderUsed ? 1 : 0, tileInterpolationUsed ? 1 : 0, lookAtInterpolationUsed ? 1 : 0);
                    matchingProfiler.end();
                    matchingProfiler.log();

                    const bool displayRateAboveOriginal = (workload.viOriginalRate > 0) && (workloadConfig.targetRate > workload.viOriginalRate);
                    generateInterpolatedFrames = !workload.paused && displayRateAboveOriginal && !interpolationTargetKey.isEmpty();

                    const bool resetTicks = !generateInterpolatedFrames || (originalRateForTicks != workload.viOriginalRate) || (displayRateForTicks != workloadConfig.targetRate) || !displayRateAboveOriginal;
                    if (resetTicks) {
                        logicalTicks = 0;
                        displayTicks = 0;
                        originalRateForTicks = workload.viOriginalRate;
                        displayRateForTicks = workloadConfig.targetRate;
                    }
                }

                // Estimate amount of frames to render based on how many display frames it'd take to reach the next logical frame.
                uint32_t displayFrames = 1;
                if (generateInterpolatedFrames) {
                    logicalTicks += workloadConfig.targetRate;
                    displayFrames = uint32_t((logicalTicks - displayTicks) / workload.viOriginalRate);
                    deltaTimeMs = 1.0f / float(workloadConfig.targetRate);

                    if ((displayFrames > 1) && frameReduction) {
                        displayTicks += workload.viOriginalRate;
                        displayFrames--;
                        frameReduction = false;
                    }

                    assert((logicalTicks > displayTicks) && "Logical ticks must always remain bigger than the display ticks.");
                    assert(((logicalTicks - displayTicks) <= (workloadConfig.targetRate + workload.viOriginalRate)) && "The gap between logical ticks and display ticks can't be bigger than the target rate.");
                    assert((displayFrames > 0) && "At least one display frame must be generated.");
                }
                else if (workload.viOriginalRate > 0) {
                    deltaTimeMs = 1.0f / float(workload.viOriginalRate);
                }

                ext.sharedResources->viOriginalRate = workload.viOriginalRate;
                
                // Get the current and previous set of frame counters. The other set can be in use by the present queue. Skip if no new present event has arrived before this workload event.
                InterpolatedFrameCounters &prevFrameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
                const bool useDifferentCounters = (lastPresentId != workload.presentId);
                if (useDifferentCounters) {
                    ext.sharedResources->interpolatedFramesIndex = ext.sharedResources->interpolatedFramesIndex ^ 1;
                    lastPresentId = workload.presentId;
                }
                // If the same set of counters is used, we wait until the presentation of its targets is finished so the targets are available to use. Waiting is ignored
                // if the frame counter has never presented anything yet, as it'll only be a valid value if the previous present event actually did something.
                else if (generateInterpolatedFrames) {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                        return (prevFrameCounters.presented == 0) || (prevFrameCounters.presented >= prevFrameCounters.available);
                    });
                }

                InterpolatedFrameCounters &curFrameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
                curFrameCounters.skipped = false;
                curFrameCounters.presented = 0;
                curFrameCounters.available = 0;
                curFrameCounters.count = displayFrames;

                // Create as many render targets as required to store the interpolated targets.
                auto &interpolatedTargets = ext.sharedResources->interpolatedColorTargets;
                const bool usingMSAA = (ext.sharedResources->renderTargetManager.multisampling.sampleCount > 1);
                const bool usesHDR = ext.sharedResources->renderTargetManager.usesHDR;
                uint32_t requiredFrames = (usingMSAA && generateInterpolatedFrames) ? displayFrames : (displayFrames - 1);
                if ((requiredFrames > 0) && (interpolatedTargets.size() < requiredFrames)) {
                    uint32_t previousSize = uint32_t(interpolatedTargets.size());
                    interpolatedTargets.resize(requiredFrames);
                    for (uint32_t i = previousSize; i < requiredFrames; i++) {
                        interpolatedTargets[i] = std::make_unique<RenderTarget>(interpolationTargetKey.address, Framebuffer::Type::Color, RenderMultisampling(), usesHDR);
                    }
                }
                
                const int64_t originalTimeMicro = (workload.viOriginalRate > 0) ? (1000000 / workload.viOriginalRate) : 0;
                const int64_t setupTimeMicro = workloadTimer.elapsedMicroseconds();
                const int64_t adjustedTimeWindowMicro = originalTimeMicro - setupTimeMicro;
                const int64_t maxTimePerFrameMicro = adjustedTimeWindowMicro / displayFrames;
                bool skippedFrames = false;
                bool skipWorkloadNow = false;
                uint32_t targetIndex = 0;
                uint32_t framesRendered = 0;
                int64_t renderTimeTotalMicro = 0;
                for (uint32_t frame = 0; (frame < displayFrames) && !skipWorkloadNow; frame++) {
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop frame begin frame=%u displayFrames=%u", frame, displayFrames);
                    // Evaluate if this frame should be skipped. Measure the current time and compare it to what frame is estimated should be have been rendered by now.
                    if ((frame > 0) && (originalTimeMicro > 0)) {
                        const int64_t currentTimeMicro = workloadTimer.elapsedMicroseconds() - setupTimeMicro;
                        const int64_t expectedTimeMicro = frame * maxTimePerFrameMicro;
                        const int64_t measuredFrameMicro = renderTimeTotalMicro / framesRendered;
                        if ((currentTimeMicro > expectedTimeMicro) || ((currentTimeMicro + measuredFrameMicro) > adjustedTimeWindowMicro)) {
                            displayTicks += workload.viOriginalRate;
                            skippedFrames = true;
                            continue;
                        }
                    }

                    RenderTarget *overrideTarget = nullptr;
                    uint32_t overrideModifier = 0;
                    if (generateInterpolatedFrames) {
                        prevFrameWeight = std::clamp((workloadConfig.targetRate + displayTicks - logicalTicks) / float(workloadConfig.targetRate), 0.0f, 1.0f);
                        displayTicks += workload.viOriginalRate;
                        curFrameWeight = std::clamp((workloadConfig.targetRate + displayTicks - logicalTicks) / float(workloadConfig.targetRate), 0.0f, 1.0f);

                        // Override the render target.
                        if (usingMSAA || (frame > 0)) {
                            overrideTarget = interpolatedTargets[targetIndex].get();
                            overrideModifier = (targetIndex + 1);

                            if (useDifferentCounters && (prevFrameCounters.available > 0) && (targetIndex < prevFrameCounters.available)) {
                                // Wait until the target has finished presenting if the alternate frame counter (used by the present queue) is making use of this target.
                                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                                    frameReduction = frameReduction || (prevFrameCounters.presented <= targetIndex);
                                    return prevFrameCounters.presented > targetIndex;
                                });
                            }

                            targetIndex++;
                        }
                    }
                    else if (workload.paused) {
                        curFrameWeight = workload.debuggerRenderer.interpolationWeight;
                        prevFrameWeight = 1.0f - curFrameWeight;
                    }
                    else {
                        prevFrameWeight = 0.0f;
                        curFrameWeight = 1.0f;
                    }

                    const bool uploadExtras = (frame == 0) && workloadConfig.raytracingEnabled;
                    if (uploadExtras) {
                        BufferUploader::Upload extrasUpload = { workload.drawData.extraParams.data(), { 0, workload.drawData.extraParams.size() }, sizeof(interop::ExtraParams), RenderBufferFlag::STORAGE, {}, &workload.drawBuffers.extraParamsBuffer };
                        ext.workloadExtrasUploader->submit(ext.workloadGraphicsWorker, { extrasUpload });
                    }

                    int64_t renderTimeMicro = workloadTimer.elapsedMicroseconds();
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before threadRenderFrame frame=%u", frame);
                    threadRenderFrame(curFrame, prevFrame, workloadConfig, workload.debuggerRenderer, workload.debuggerCamera, curFrameWeight, prevFrameWeight, deltaTimeMs,
                        interpolationTargetKey, interpolationTargetFbPairIndex, overrideTarget, overrideModifier, velocityUploaderUsed, uploadExtras, tileInterpolationUsed, lookAtInterpolationUsed);
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after threadRenderFrame frame=%u", frame);

                    // Add total time the frame took to render.
                    renderTimeTotalMicro += workloadTimer.elapsedMicroseconds() - renderTimeMicro;

                    // After one frame is rendered, we indicate the workload has been processed so the present thread can start presenting frames as soon as it can.
                    if (frame == 0) {
                        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before threadAdvanceWorkloadId frame=%u id=%llu",
                            frame, static_cast<unsigned long long>(workload.workloadId));
                        threadAdvanceWorkloadId(workload.workloadId);
                        RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after threadAdvanceWorkloadId frame=%u", frame);
                    }

                    // For every additional frame, we increase the frames available and notify the present queue.
                    if (generateInterpolatedFrames && (usingMSAA || (frame > 0))) {
                        {
                            std::scoped_lock<std::mutex> cursorLock(cursorMutex);
                            skipWorkloadNow = ((frame + 1) < displayFrames) && (writeCursor != threadCursor);
                        }

                        {
                            std::scoped_lock<std::mutex> managerLock(ext.sharedResources->interpolatedMutex);
                            curFrameCounters.skipped = skipWorkloadNow;
                            curFrameCounters.available++;
                        }

                        // Add the amount of display ticks that correspond to the remaining frames.
                        if (skipWorkloadNow) {
                            displayTicks += workload.viOriginalRate * (displayFrames - (frame + 1));
                        }

                        ext.sharedResources->interpolatedCondition.notify_all();
                    }

                    framesRendered++;
                }

                // Set the skipped parameter on the frame counter if the workload wasn't skipped but some of its frames were.
                if (skippedFrames && !skipWorkloadNow) {
                    {
                        std::scoped_lock<std::mutex> managerLock(ext.sharedResources->interpolatedMutex);
                        curFrameCounters.skipped = true;
                    }

                    ext.sharedResources->interpolatedCondition.notify_all();
                }

                threadConfigurationValidate();
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after configuration validate");

                if (!workload.paused) {
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop before advance barrier");
                    threadAdvanceBarrier();
                    RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop after advance barrier");
                }

                processCursor = -1;
                workloadProfiler.end();
                workloadProfiler.log();
                workloadProfiler.reset();
                RT64_WORKLOAD_ANDROID_LOG("WorkloadQueue::renderThreadLoop workload complete");
            }
        }
    }
    
    void WorkloadQueue::idleThreadLoop() {
        // Beware traveler as you enter the zone of dirty driver hacks. Given N64 games are not exactly a demanding thing to render
        // nowadays for modern GPUs and due to how the plugin's cooperative multiqueue system works, it's sometimes just not possible
        // to keep the GPU busy at all times. It is often the case that the GPU might've already rendered all the frames it needed to
        // generate before the screen update event from the emulator even arrives on time.
        // 
        // Under this situation, some drivers are a bit too trigger-happy to downclock the GPU and lower the power consumption, eventually
        // resulting in very low power states that cause unwanted frametime spikes that can no longer reach the target framerate. This
        // results in visible judder during gameplay.
        //
        // This thread will take care of sending some GPU work that does nothing useful while the GPU is not actually busy generating
        // new frames. The waiting interval is close to the minimum resolution the OS provides and big enough to not cause any significant
        // delays or unwanted power consumption: it's just enough to keep the driver from downclocking to a power state level that is
        // usually intended for 2D work or video playback.
        //
        // This workaround is not required if the driver is configured to be at the "Max Performance" power state.

        Thread::setCurrentThreadName("RT64 Idle");

        const ShaderRecord &idle = ext.shaderLibrary->idle;
        RenderCommandList *commandList = ext.workloadGraphicsWorker->commandList.get();
        while (threadsRunning) {
            {
                std::unique_lock<std::mutex> idleLock(idleMutex);
                idleCondition.wait(idleLock, [&]() {
                    return idleActive || !threadsRunning;
                });
            }

            if (threadsRunning) {
                if (workerMutex.try_lock()) {
                    commandList->begin();
                    commandList->setPipeline(idle.pipeline.get());
                    commandList->setComputePipelineLayout(idle.pipelineLayout.get());
                    commandList->dispatch(1, 1, 1);
                    commandList->end();
                    ext.workloadGraphicsWorker->execute();
                    ext.workloadGraphicsWorker->wait();
                    workerMutex.unlock();
                }
                
                Thread::sleepMilliseconds(1);
            }
        }
    }
};
