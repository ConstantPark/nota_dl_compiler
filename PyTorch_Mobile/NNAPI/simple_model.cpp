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
#include "simple_model.h"

#include 
#include 
#include 
#include 
#include 

/**
 * SimpleModel Constructor.
 *
 * Initialize the member variables, including the shared memory objects.
 */
SimpleModel::SimpleModel(size_t size, int protect, int fd, size_t offset) :
        model_(nullptr),
        compilation_(nullptr),
        dimLength_(TENSOR_SIZE),
        modelDataFd_(fd),
        offset_(offset) {
    tensorSize_ = dimLength_;
    inputTensor1_.resize(tensorSize_);

    // Create ANeuralNetworksMemory from a file containing the trained data.
    int32_t status = ANeuralNetworksMemory_createFromFd(size + offset, protect, fd, 0,
                                                        &memoryModel_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for trained weights");
        return;
    }

    // Create ASharedMemory to hold the data for the second input tensor and output output tensor.
    inputTensor2Fd_ = ASharedMemory_create("input2", tensorSize_ * sizeof(float));
    outputTensorFd_ = ASharedMemory_create("output", tensorSize_ * sizeof(float));

    // Create ANeuralNetworksMemory objects from the corresponding ASharedMemory objects.
    status = ANeuralNetworksMemory_createFromFd(tensorSize_ * sizeof(float),
                                                PROT_READ,
                                                inputTensor2Fd_, 0,
                                                &memoryInput2_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for Input2");
        return;
    }
    status = ANeuralNetworksMemory_createFromFd(tensorSize_ * sizeof(float),
                                                PROT_READ | PROT_WRITE,
                                                outputTensorFd_, 0,
                                                &memoryOutput_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for Output");
        return;
    }
}

/**
 * Create a graph that consists of three operations: two additions and a
 * multiplication.
 * The sums created by the additions are the inputs to the multiplication. In
 * essence, we are creating a graph that computes:
 *        (tensor0 + tensor1) * (tensor2 + tensor3).
 *
 * tensor0 ---+
 *            +--- ADD ---> intermediateOutput0 ---+
 * tensor1 ---+                                    |
 *                                                 +--- MUL---> output
 * tensor2 ---+                                    |
 *            +--- ADD ---> intermediateOutput1 ---+
 * tensor3 ---+
 *
 * Two of the four tensors, tensor0 and tensor2 being added are constants, defined in the
 * model. They represent the weights that would have been learned during a training process.
 *
 * The other two tensors, tensor1 and tensor3 will be inputs to the model. Their values will be
 * provided when we execute the model. These values can change from execution to execution.
 *
 * Besides the two input tensors, an optional fused activation function can
 * also be defined for ADD and MUL. In this example, we'll simply set it to NONE.
 *
 * The graph then has 10 operands:
 *  - 2 tensors that are inputs to the model. These are fed to the two
 *      ADD operations.
 *  - 2 constant tensors that are the other two inputs to the ADD operations.
 *  - 1 fuse activation operand reused for the ADD operations and the MUL operation.
 *  - 2 intermediate tensors, representing outputs of the ADD operations and inputs to the
 *      MUL operation.
 *  - 1 model output.
 *
 * @return true for success, false otherwise
 */
bool SimpleModel::CreateCompiledModel() {
    int32_t status;

    // Create the ANeuralNetworksModel handle.
    // ①Create()：ANeuralNetworksModelオブジェクトを生成し、変数「model」に格納する
    status = ANeuralNetworksModel_create(&model_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_create failed");
        return false;
    }

    uint32_t dimensions[] = {dimLength_};
    ANeuralNetworksOperandType float32TensorType{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    ANeuralNetworksOperandType scalarInt32Type{
            .type = ANEURALNETWORKS_INT32,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    /**
     * Add operands and operations to construct the model.
     *
     * Operands are implicitly identified by the order in which they are added to the model,
     * starting from 0.
     *
     * These indexes are not returned by the model_addOperand call. The application must
     * manage these values. Here, we use opIdx to do the bookkeeping.
     */
    uint32_t opIdx = 0;

    // We first add the operand for the NONE activation function, and set its
    // value to ANEURALNETWORKS_FUSED_NONE.
    // This constant scalar operand will be used for all 3 operations.
    // ②Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド1(fusedActivationFuncNone)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &scalarInt32Type);
    uint32_t fusedActivationFuncNone = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            fusedActivationFuncNone);
        return false;
    }

    FuseCode fusedActivationCodeValue = ANEURALNETWORKS_FUSED_NONE;
    status = ANeuralNetworksModel_setOperandValue(
            model_, fusedActivationFuncNone, &fusedActivationCodeValue,
            sizeof(fusedActivationCodeValue));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_setOperandValue failed for operand (%d)",
                            fusedActivationFuncNone);
        return false;
    }

    // Add operands for the tensors.
    // ③Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド2(tensor0)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor0 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor0);
        return false;
    }
    // tensor0 is a constant tensor that was established during training.
    // We read these values from the corresponding ANeuralNetworksMemory object.
    status = ANeuralNetworksModel_setOperandValueFromMemory(model_,
                                                            tensor0,
                                                            memoryModel_,
                                                            offset_,
                                                            tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_setOperandValueFromMemory failed for operand (%d)",
                            tensor0);
        return false;
    }

    // tensor1 is one of the user provided input tensors to the trained model.
    // Its value is determined pre-execution.
    // ④Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド3(tensor1)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor1 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor1);
        return false;
    }

    // tensor2 is a constant tensor that was established during training.
    // We read these values from the corresponding ANeuralNetworksMemory object.
    // ⑤Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド4(tensor2)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor2 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor2);
        return false;
    }
    status = ANeuralNetworksModel_setOperandValueFromMemory(
            model_, tensor2, memoryModel_, offset_ + tensorSize_ * sizeof(float),
            tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_setOperandValueFromMemory failed for operand (%d)",
                            tensor2);
        return false;
    }

    // tensor3 is one of the user provided input tensors to the trained model.
    // Its value is determined pre-execution.
    // ⑥Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド5(tensor3)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor3 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor3);
        return false;
    }

    // intermediateOutput0 is the output of the first ADD operation.
    // Its value is computed during execution.
    // ⑦Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド6(intermediateOutput0)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t intermediateOutput0 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            intermediateOutput0);
        return false;
    }

    // intermediateOutput1 is the output of the second ADD operation.
    // Its value is computed during execution.
    // ⑧Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド7(intermediateOutput1)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t intermediateOutput1 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            intermediateOutput1);
        return false;
    }

    // multiplierOutput is the output of the MUL operation.
    // Its value will be computed during execution.
    // ⑨Create()：ANeuralNetworksModelオブジェクトの「model_」にオペランド8(multiplierOutput)を格納する
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t multiplierOutput = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            multiplierOutput);
        return false;
    }

    // Add the first ADD operation.
    std::vector add1InputOperands = {
            tensor0,
            tensor1,
            fusedActivationFuncNone,
    };
    // ⑩Create()：ANeuralNetworksModelオブジェクトの「model_」にaddオペレーション1(add1InputOperands)を格納する
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_ADD,
                                               add1InputOperands.size(), add1InputOperands.data(),
                                               1, &intermediateOutput0);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperation failed for ADD_1");
        return false;
    }

    // Add the second ADD operation.
    // Note the fusedActivationFuncNone is used again.
    std::vector add2InputOperands = {
            tensor2,
            tensor3,
            fusedActivationFuncNone,
    };
    // ⑪Create()：ANeuralNetworksModelオブジェクトの「model_」にaddオペレーション(add2InputOperands)を格納する
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_ADD,
                                               add2InputOperands.size(),add2InputOperands.data(),
                                               1, &intermediateOutput1);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperation failed for ADD_2");
        return false;
    }

    // Add the MUL operation.
    // Note that intermediateOutput0 and intermediateOutput1 are specified
    // as inputs to the operation.
    std::vector mulInputOperands = {
            intermediateOutput0,
            intermediateOutput1,
            fusedActivationFuncNone};
    // ⑫Create()：ANeuralNetworksModelオブジェクトの「model_」にmulオペレーション(mulInputOperands)を格納する
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_MUL,
                                               mulInputOperands.size(),mulInputOperands.data(),
                                               1, &multiplierOutput);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperation failed for MUL");
        return false;
    }

    // Identify the input and output tensors to the model.
    // Inputs: {tensor1, tensor3}
    // Outputs: {multiplierOutput}
    std::vector modelInputOperands = {
            tensor1, tensor3,
    };
    // ⑬Create()：ANeuralNetworksModelオブジェクトの「model_」にidオペレーション(modelInputOperands)を格納する
    status = ANeuralNetworksModel_identifyInputsAndOutputs(model_,
                                                           modelInputOperands.size(),
                                                           modelInputOperands.data(),
                                                           1,
                                                           &multiplierOutput);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_identifyInputsAndOutputs failed");
        return false;
    }

    // Finish constructing the model.
    // The values of constant and intermediate operands cannot be altered after
    // the finish function is called.
    // ⑭Create()：ANeuralNetworksModelオブジェクトの「model_」の構成を完了する
    status = ANeuralNetworksModel_finish(model_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_finish failed");
        return false;
    }

    // Create the ANeuralNetworksCompilation object for the constructed model.
    // ⑮Create()：ANeuralNetworksModelオブジェクトの「model_」のコンパイルを完了する
    status = ANeuralNetworksCompilation_create(model_, &compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_create failed");
        return false;
    }

    // Set the preference for the compilation, so that the runtime and drivers
    // can make better decisions.
    // Here we prefer to get the answer quickly, so we choose
    // ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER.
    // ⑯Create()：ANeuralNetworksModelオブジェクトの「model_」のプレファレンスを設定する
    status = ANeuralNetworksCompilation_setPreference(compilation_,
                                                      ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_setPreference failed");
        return false;
    }

    // Finish the compilation.
    // ⑰Create()：ANeuralNetworksModelオブジェクトの「model_」のモデルの定義の完了を設定する
    status = ANeuralNetworksCompilation_finish(compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_finish failed");
        return false;
    }

    return true;
}

/**
 * Compute with the given input data.
 * @param modelInputs:
 *    inputValue1:   The values to fill tensor1
 *    inputValue2:   The values to fill tensor3
 * @return  computed result, or 0.0f if there is error.
 */
bool SimpleModel::Compute(float inputValue1, float inputValue2,
                          float *result) {
    if (!result) {
        return false;
    }

    // Create an ANeuralNetworksExecution object from the compiled model.
    // Note:
    //   1. All the input and output data are tied to the ANeuralNetworksExecution object.
    //   2. Multiple concurrent execution instances could be created from the same compiled model.
    // This sample only uses one execution of the compiled model.
    // ①Compute()：ANeuralNetworksExecutionオブジェクトを生成し、変数「execution」に格納する
    ANeuralNetworksExecution *execution;
    int32_t status = ANeuralNetworksExecution_create(compilation_, &execution);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_create failed");
        return false;
    }

    // Set all the elements of the first input tensor (tensor1) to the same value as inputValue1.
    // It's not a realistic example but it shows how to pass a small tensor
    // to an execution.
    // ②Compute()：「inputTensor1」に「inputValue1」の値を入れる
    std::fill(inputTensor1_.data(), inputTensor1_.data() + tensorSize_,
              inputValue1);

    // Tell the execution to associate inputTensor1 to the first of the two model inputs.
    // Note that the index "0" here means the first operand of the modelInput list
    // {tensor1, tensor3}, which means tensor1.
    // ③Compute()：「inputTensor1」を「excution」クラスに設定する
    status = ANeuralNetworksExecution_setInput(execution, 0, nullptr,
                                               inputTensor1_.data(),
                                               tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_setInput failed for input1");
        return false;
    }

    // Set the values of the the second input operand (tensor3) to be inputValue2.
    // In reality, the values in the shared memory region will be manipulated by
    // other modules or processes.
    // ④Compute()：「inputTensor1」を「excution」クラスに設定する
    float *inputTensor2Ptr = reinterpret_cast(mmap(nullptr, tensorSize_ * sizeof(float),
                                                            PROT_READ | PROT_WRITE, MAP_SHARED,
                                                            inputTensor2Fd_, 0));
    for (int i = 0; i < tensorSize_; i++) {
        *inputTensor2Ptr = inputValue2;
        inputTensor2Ptr++;
    }
    munmap(inputTensor2Ptr, tensorSize_ * sizeof(float));

    // ANeuralNetworksExecution_setInputFromMemory associates the operand with a shared memory
    // region to minimize the number of copies of raw data.
    // Note that the index "1" here means the second operand of the modelInput list
    // {tensor1, tensor3}, which means tensor3.
    // ⑤Compute()：「ANeuralNetworksExecution_setInputFromMemory()」関数で入力値を指定する
    status = ANeuralNetworksExecution_setInputFromMemory(execution, 1, nullptr,
                                                         memoryInput2_, 0,
                                                         tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_setInputFromMemory failed for input2");
        return false;
    }

    // Set the output tensor that will be filled by executing the model.
    // We use shared memory here to minimize the copies needed for getting the output data.
    // ⑥Compute()：「ANeuralNetworksExecution_setOutputFromMemory()」関数で出力先を指定する
    status = ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
                                                          memoryOutput_, 0,
                                                          tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_setOutputFromMemory failed for output");
        return false;
    }

    // Start the execution of the model.
    // Note that the execution here is asynchronous, and an ANeuralNetworksEvent object will be
    // created to monitor the status of the execution.
    // ⑦Compute()：「ANeuralNetworksExecution_startCompute()」関数で推論を実行する
    ANeuralNetworksEvent *event = nullptr;
    status = ANeuralNetworksExecution_startCompute(execution, &event);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_startCompute failed");
        return false;
    }

    // Wait until the completion of the execution. This could be done on a different
    // thread. By waiting immediately, we effectively make this a synchronous call.
    // ⑧Compute()：lNetworksEvent__wait()」関数で推論を実行の完了を待機する
    status = ANeuralNetworksEvent_wait(event);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksEvent_wait failed");
        return false;
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    // Validate the results.
    const float goldenRef = (inputValue1 + 0.5f) * (inputValue2 + 0.5f);
    float *outputTensorPtr = reinterpret_cast(mmap(nullptr,
                                                            tensorSize_ * sizeof(float),
                                                            PROT_READ, MAP_SHARED,
                                                            outputTensorFd_, 0));
    for (int32_t idx = 0; idx < tensorSize_; idx++) {
        float delta = outputTensorPtr[idx] - goldenRef;
        delta = (delta < 0.0f) ? (-delta) : delta;
        if (delta > FLOAT_EPISILON) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "Output computation Error: output0(%f), delta(%f) @ idx(%d)",
                                outputTensorPtr[0], delta, idx);
        }
    }
    *result = outputTensorPtr[0];
    munmap(outputTensorPtr, tensorSize_ * sizeof(float));
    return result;
}

/**
 * SimpleModel Destructor.
 *
 * Release NN API objects and close the file descriptors.
 */
SimpleModel::~SimpleModel() {
    ANeuralNetworksCompilation_free(compilation_);
    ANeuralNetworksModel_free(model_);
    ANeuralNetworksMemory_free(memoryModel_);
    ANeuralNetworksMemory_free(memoryInput2_);
    ANeuralNetworksMemory_free(memoryOutput_);
    close(inputTensor2Fd_);
    close(outputTensorFd_);
    close(modelDataFd_);
}
