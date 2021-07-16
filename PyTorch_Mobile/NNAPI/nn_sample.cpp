/**
 * Copyright 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include 
#include 
#include 
#include 
#include 

#include 
#include 
#include 
#include 

#include "simple_model.h"

//LOGV()を表示するため(これを記述しなくてもLOGE, LOGWなどは表示される)
//#define LOG_NDEBUG 0
//logcatに出力されるタグ(必須)
//#define LOG_TAG "nn_sample.cpp"
//ログを出力するマクロを定義するヘッダーファイル(必須)
//#include 

extern "C"
JNIEXPORT jlong
JNICALL
Java_com_hiroict_talkai001_TestNeuralNetworksActivity_initModel(
        JNIEnv *env,
        jobject /* this */,
        jobject _assetManager,
        jstring _assetName) {
    // Get the file descriptor of the the model data file.
    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    const char *assetName = env->GetStringUTFChars(_assetName, NULL);
    AAsset *asset = AAssetManager_open(assetManager, assetName, AASSET_MODE_BUFFER);
    if(asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        return 0;
    }
    env->ReleaseStringUTFChars(_assetName, assetName);
    off_t offset, length;
    int fd = AAsset_openFileDescriptor(asset, &offset, &length);
    AAsset_close(asset);
    if (fd < 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to open the model_data file descriptor.");
        return 0;
    }
    SimpleModel* nn_model = new SimpleModel(length, PROT_READ, fd, offset);
    if (!nn_model->CreateCompiledModel()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to prepare the model.");
        return 0;
    }

    return (jlong)(uintptr_t)nn_model;
}

extern "C"
JNIEXPORT jfloat
JNICALL
Java_com_hiroict_talkai001_TestNeuralNetworksActivity_startCompute(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel,
        jfloat inputValue1,
        jfloat inputValue2) {
    SimpleModel* nn_model = (SimpleModel*) _nnModel;
    float result = 0.0f;
    nn_model->Compute(inputValue1, inputValue2, &result);
    __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,"計算結果： %5.2f",result);
    return result;
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_hiroict_talkai001_TestNeuralNetworksActivity_destroyModel(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel) {
    SimpleModel* nn_model = (SimpleModel*) _nnModel;
    delete(nn_model);
}
