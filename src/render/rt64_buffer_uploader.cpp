//
// RT64
//

#include <algorithm>
#include <cstring>

#include "common/rt64_thread.h"

#include "rt64_buffer_uploader.h"

#if defined(__ANDROID__)
#include <android/log.h>
#define RT64_ANDROID_BUFFER_LOG(...) ((void)0)
#else
#define RT64_ANDROID_BUFFER_LOG(...)
#endif

namespace RT64 {
    // Common functions.

    static uint64_t roundUp(uint64_t value, uint64_t powerOf2Alignment) {
        return (value + powerOf2Alignment - 1) & ~(powerOf2Alignment - 1);
    }

    // BufferUploader::Upload

    bool BufferUploader::Upload::valid() const {
        return (srcData != nullptr) && (srcDataIndexRange.second > srcDataIndexRange.first);
    }

    // BufferUploader

    BufferUploader::BufferUploader(RenderDevice *device) {
        assert(device != nullptr);

        this->device = device;
        workAvailable = false;
        thread = new std::thread(&BufferUploader::threadLoop, this);
    }

    BufferUploader::~BufferUploader() {
        running = false;
        workCondition.notify_all();
        thread->join();
        delete thread;
    }

    void BufferUploader::threadLoop() {
        Thread::setCurrentThreadName("RT64 Buffer");

        running = true;

        while (running) {
            std::unique_lock<std::mutex> queueLock(workMutex);
            workCondition.wait(queueLock, [this]() {
                return !running || workAvailable;
            });
            
            if (running) {
                for (const Upload &u : pendingUploads) {
                    threadUpload(u);
                }
            }

            {
                std::unique_lock<std::mutex> readyLock(readyMutex);
                workAvailable = false;
            }

            readyCondition.notify_all();
        }
    }

    void BufferUploader::threadUpload(const Upload &upload) {
        if (!upload.valid()) {
            return;
        }

        assert(upload.dstPair != nullptr);
        const size_t srcOffset = upload.srcDataIndexRange.first * upload.srcDataStride;
        const size_t srcSize = (upload.srcDataIndexRange.second - upload.srcDataIndexRange.first) * upload.srcDataStride;
        const RenderRange writtenRange(srcOffset, srcOffset + srcSize);
        RT64_ANDROID_BUFFER_LOG("BufferUploader::threadUpload begin range=(%zu,%zu) stride=%zu bytes=%zu allocated=%llu",
            upload.srcDataIndexRange.first, upload.srcDataIndexRange.second, upload.srcDataStride, srcSize,
            static_cast<unsigned long long>(upload.dstPair->allocatedSize));
        uint8_t *dstData = static_cast<uint8_t *>(upload.dstPair->uploadBuffer->map());
        RT64_ANDROID_BUFFER_LOG("BufferUploader::threadUpload after map bytes=%zu", srcSize);
        memcpy(dstData + srcOffset, static_cast<const uint8_t *>(upload.srcData) + srcOffset, srcSize);
        RT64_ANDROID_BUFFER_LOG("BufferUploader::threadUpload after memcpy bytes=%zu", srcSize);
        upload.dstPair->uploadBuffer->unmap(0, &writtenRange);
        RT64_ANDROID_BUFFER_LOG("BufferUploader::threadUpload end bytes=%zu", srcSize);
    }

    void BufferUploader::updateResources(RenderWorker *worker, std::vector<Upload> &blankUploads) {
        uint32_t uploadIndex = 0;
        for (Upload &u : blankUploads) {
            // Ignore the reallocation of the buffer if the required size is already enough. We always create a buffer if it hasn't been created yet.
            const size_t requiredSize = u.srcDataIndexRange.second * u.srcDataStride;
            BufferPair &bufferPair = *u.dstPair;
            RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] valid=%d range=(%zu,%zu) stride=%zu required=%zu allocated=%llu flags=0x%X views=%zu",
                uploadIndex, int(u.valid()), u.srcDataIndexRange.first, u.srcDataIndexRange.second, u.srcDataStride, requiredSize,
                static_cast<unsigned long long>(bufferPair.allocatedSize), uint32_t(u.bufferFlags), u.formatViews.size());
            if ((bufferPair.defaultBuffer != nullptr) && (!u.valid() || (bufferPair.allocatedSize >= requiredSize))) {
                RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] reuse", uploadIndex);
                uploadIndex++;
                continue;
            }

            bufferPair.defaultViews.clear();
            
            // Recreate the buffer pair.
            const uint64_t BlockAlignment = 256;
            bufferPair.allocatedSize = std::max(uint64_t((requiredSize * 3) / 2), BlockAlignment);
            bufferPair.allocatedSize = roundUp(bufferPair.allocatedSize, BlockAlignment);
            RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] before uploadBuffer create allocated=%llu",
                uploadIndex, static_cast<unsigned long long>(bufferPair.allocatedSize));
            bufferPair.uploadBuffer = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(bufferPair.allocatedSize));
            RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] after uploadBuffer create", uploadIndex);
            bufferPair.defaultBuffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(bufferPair.allocatedSize, u.bufferFlags));
            RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] after defaultBuffer create", uploadIndex);

            bufferPair.defaultViews.reserve(u.formatViews.size());
            uint32_t viewIndex = 0;
            for (RenderFormat format : u.formatViews) {
                RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] before view[%u] format=%u",
                    uploadIndex, viewIndex, uint32_t(format));
                bufferPair.defaultViews.emplace_back(bufferPair.defaultBuffer->createBufferFormattedView(format));
                RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] after view[%u]", uploadIndex, viewIndex);
                viewIndex++;
            }

            // Since the buffers had to be recreated, reupload all the data by modifying the source upload.
            u.srcDataIndexRange.first = 0;
            RT64_ANDROID_BUFFER_LOG("BufferUploader::updateResources upload[%u] recreated", uploadIndex);
            uploadIndex++;
        }
    }

    void BufferUploader::submit(RenderWorker *worker, const std::vector<Upload> &uploads) {
        RT64_ANDROID_BUFFER_LOG("BufferUploader::submit begin uploads=%zu", uploads.size());
        {
            std::unique_lock<std::mutex> queueLock(workMutex);
            RT64_ANDROID_BUFFER_LOG("BufferUploader::submit after lock");
            pendingUploads = uploads;
            RT64_ANDROID_BUFFER_LOG("BufferUploader::submit after copy");
            updateResources(worker, pendingUploads);
            RT64_ANDROID_BUFFER_LOG("BufferUploader::submit after updateResources");
            workAvailable = true;
        }

        workCondition.notify_all();
        RT64_ANDROID_BUFFER_LOG("BufferUploader::submit end");
    }

    void BufferUploader::commandListBeforeBarriers(RenderWorker *worker) {
        thread_local std::vector<RenderBufferBarrier> beforeBarriers;
        beforeBarriers.clear();

        for (const Upload &u : pendingUploads) {
            if (!u.valid()) {
                continue;
            }

            auto &defaultBuffer = u.dstPair->defaultBuffer;
            beforeBarriers.push_back(RenderBufferBarrier(defaultBuffer.get(), RenderBufferAccess::WRITE));
        }

        if (!beforeBarriers.empty()) {
            worker->commandList->barriers(RenderBarrierStage::COPY, beforeBarriers);
        }
    }

    void BufferUploader::commandListCopyResources(RenderWorker *worker) {
        for (const Upload &u : pendingUploads) {
            if (!u.valid()) {
                continue;
            }

            const uint64_t srcOffset = u.srcDataIndexRange.first * u.srcDataStride;
            const uint64_t srcSize = (u.srcDataIndexRange.second - u.srcDataIndexRange.first) * u.srcDataStride;
            worker->commandList->copyBufferRegion(u.dstPair->defaultBuffer->at(srcOffset), u.dstPair->uploadBuffer->at(srcOffset), srcSize);
        }
    }

    void BufferUploader::commandListAfterBarriers(RenderWorker *worker) {
        thread_local std::vector<RenderBufferBarrier> afterBarriers;
        afterBarriers.clear();

        for (const Upload &u : pendingUploads) {
            if (!u.valid()) {
                continue;
            }

            auto &defaultBuffer = u.dstPair->defaultBuffer;
            afterBarriers.push_back(RenderBufferBarrier(defaultBuffer.get(), RenderBufferAccess::READ));
        }

        if (!afterBarriers.empty()) {
            worker->commandList->barriers(RenderBarrierStage::ALL, afterBarriers);
        }
    }
    
    void BufferUploader::wait() {
        std::unique_lock<std::mutex> readyLock(readyMutex);
        readyCondition.wait(readyLock, [this]() {
            return !workAvailable;
        });
    }
};
