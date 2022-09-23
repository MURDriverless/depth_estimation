#include "EntropyCalibrator.hpp"
#include <fstream>
#include <iterator>
#include <cassert>
#include <string.h>
#include <algorithm>

namespace nvinfer1
{
    Int8EntropyCalibrator::Int8EntropyCalibrator(int BatchSize,const std::vector<std::vector<float>>& data,
                                            const std::string& CalibDataName /*= ""*/,bool readCache /*= true*/)
        : mCalibDataName(CalibDataName),mBatchSize(BatchSize),mReadCache(readCache)
    {     
	if (data.size() > 0)
	{
            mDatas.reserve(data.size());
            mDatas = data;

            mInputCount =  BatchSize * data[0].size();
            mCurBatchData = new float[mInputCount];
            mCurBatchIdx = 0;
            CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
	}
    }


    Int8EntropyCalibrator::~Int8EntropyCalibrator()
    {
        CUDA_CHECK(cudaFree(mDeviceInput));
        if(mCurBatchData)
            delete[] mCurBatchData;
    }


    bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (mCurBatchIdx + mBatchSize > int(mDatas.size()))
	    return false;

        float* ptr = mCurBatchData;
        size_t imgSize = mInputCount / mBatchSize;
        auto iter = mDatas.begin() + mCurBatchIdx;

        std::for_each(iter, iter + mBatchSize, [=,&ptr](std::vector<float>& val){
            assert(imgSize == val.size());
            memcpy(ptr,val.data(),imgSize*sizeof(float));
            ptr += imgSize;
        });

        CUDA_CHECK(cudaMemcpy(mDeviceInput, mCurBatchData, mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = mDeviceInput;
        mCurBatchIdx += mBatchSize;
        return true;
    }

    const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibDataName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibDataName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

}
