#include "TComTF.h"
#if USE_TENSORFLOW
TComTF::TComTF() {}
TComTF::~TComTF() {}

Status TComTF::LoadGraph(const string& graph_file_name,
	std::unique_ptr<tensorflow::Session>* session) {
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}
	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}


Tensor TComTF::ReadTensorFromYuv(Pel* piSrc, Int inputHeight, Int inputWidth, Int iStride)
{
	std::vector<float> inputYuv;
	for (int i = 0;i < inputHeight;i++)
	{
		for (int j = 0;j < inputWidth;j++)
		{
			inputYuv.push_back(piSrc[j]/ PixelRangeDouble);
		}
		piSrc += iStride;
	}
	auto mapped_X_ = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>
		(&inputYuv[0], 1, inputHeight, inputWidth,1);
	auto eigen_X_ = Eigen::Tensor<float, 4, Eigen::RowMajor>(mapped_X_);
	
	Tensor out_tensors(DT_FLOAT, TensorShape({ 1,inputHeight, inputWidth,1 }));
	out_tensors.tensor<float, 4>() = eigen_X_;
	return out_tensors;
}
Status TComTF::Tensor2Yuv(Pel* piSrc, Int inputHeight, Int inputWidth, Int iStride, Tensor inputTensors)
{
	tensorflow::TTypes<float>::Flat flatTensor = inputTensors.flat<float>();
	for (int i = 0;i < inputHeight;i++)
	{
		for (int j = 0;j < inputWidth;j++)
		{
			piSrc[j]=Clip3(0, PixelRangeInt,int(flatTensor(i*inputWidth+j)*PixelRangeDouble +0.5));
		}
		piSrc += iStride;
	}
	return Status::OK();
}
Int TComTF::setenv(const char *name, const char *value, int overwrite)
{
	int errcode = 0;
	if (!overwrite) {
		size_t envsize = 0;
		errcode = getenv_s(&envsize, NULL, 0, name);
		if (errcode || envsize) return errcode;
	}
	return _putenv_s(name, value);
}

Void TComTF::TFNetForward(const string graph_path, char* cGPUid, Pel* piSrc,Int inputHeight, Int inputWidth,Int iStride, string inputLayer, string outputLayer) {
	
	setenv("CUDA_VISIBLE_DEVICES", cGPUid, 1);

	std::unique_ptr<tensorflow::Session> session;
	
	TF_CHECK_OK(LoadGraph(graph_path, &session));
	
	Tensor inputTensor;
	inputTensor=ReadTensorFromYuv(piSrc, inputHeight, inputWidth, iStride);
	
	std::vector<Tensor> outputs;
	TF_CHECK_OK(session->Run({ { inputLayer, inputTensor } }, { outputLayer }, {}, &outputs));
	
	Tensor Y_ = outputs[0];
	TF_CHECK_OK(Tensor2Yuv(piSrc, inputHeight, inputWidth, iStride, Y_));
	session->Close();
}

#endif
#ifdef USE_TENSORFLOW_ALTER
TComTF::TComTF() {}
TComTF::~TComTF() {}

void free_buffer(void* data, size_t length) {
  free(data);
}

void deallocator(void* ptr, size_t len, void* arg) {
  free((void*) ptr);
}

Int TComTF::ReadRawTensorFromYuv(float* rawTensor, Pel* piSrc, Int inputHeight, Int inputWidth, Int iStride)
{
	for (int i = 0;i < inputHeight;i++)
	{
		for (int j = 0;j < inputWidth;j++)
		{
			rawTensor[j+i*inputWidth] = piSrc[j];
		}
		piSrc += iStride;
	}
	return 0;
}

Int TComTF::RawTensor2Yuv(Pel* piSrc, float* rawTensor, Int inputHeight, Int inputWidth, Int iStride)
{
	for (int i = 0;i < inputHeight;i++)
	{
		for (int j = 0;j < inputWidth;j++)
		{
			piSrc[j] = Clip3(0, PixelRangeInt, int(rawTensor[i*inputWidth+j]*PixelRangeDouble + 0.5));
		}
		piSrc += iStride;
	}
	return 0;
}

Int TComTF::setenv(const char *name, const char *value, int overwrite)
{
	char* exsiting_env=NULL;
	if (!overwrite) {
		exsiting_env = getenv(name);
		if (exsiting_env) return 1;
	}
	return setenv(name, value, overwrite);
}

TF_Buffer* TComTF::ReadPBFromFile(const char* file) {
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  //same as rewind(f);

    void* data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}

Void TComTF::TFNetForward(const string graph_path, char* cGPUid, Pel* piSrc, Int inputHeight, Int inputWidth, Int iStride, string inputLayer, string outputLayer) {
	
	setenv("CUDA_VISIBLE_DEVICES", cGPUid, 1);
	TF_Session* sess;
	TF_Graph* graph;
	TF_Status* status;
	graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_Buffer* graph_def = ReadPBFromFile(graph_path.c_str());
    
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
        return;
    }
    fprintf(stdout, "Successfully imported graph\n");



    // create session
    // ================================================================================
    TF_SessionOptions* opt = TF_NewSessionOptions();
    // char s[] = {16, 1, 40, 1, 50, 4, 32, 1, 64, 1};
    // TF_SetConfig(opt, s, 10, status);
    // if (TF_GetCode(status) != TF_OK) {
    //     fprintf(stderr, "ERROR: Failed setting parallelism %s\n", TF_Message(status));
    //     return;
    // }
    sess = TF_NewSession(graph, opt, status);
    TF_DeleteSessionOptions(opt);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status));
        return;
    }
    fprintf(stdout, "Successfully created session\n");
    // ================================================================================

    TF_Operation *input_op = TF_GraphOperationByName(graph, inputLayer.c_str());
    float* raw_input_data = (float*)malloc(inputHeight*inputWidth * sizeof(float));
    ReadRawTensorFromYuv(raw_input_data, piSrc, inputHeight, inputWidth, iStride);
    int64_t* raw_input_dims = (int64_t*)malloc(4 * sizeof(int64_t));
    raw_input_dims[0] = 1;
    raw_input_dims[1] = inputHeight;
    raw_input_dims[2] = inputWidth;
    raw_input_dims[3] = 1;

    /*
    TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
      TF_DataType,
      const int64_t* dims, int num_dims,
      void* data, size_t len,
      void (*deallocator)(void* data, size_t len, void* arg),
      void* deallocator_arg);
    */
    // prepare inputs
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,
                                         raw_input_dims, 4,
                                         raw_input_data, inputHeight*inputWidth * sizeof(float),
                                         deallocator,
                                         NULL
                                        );


    TF_Output* run_inputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
    run_inputs[0].oper = input_op;
    run_inputs[0].index = 0;

    TF_Tensor** run_inputs_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
    run_inputs_tensors[0] = input_tensor;

    // prepare outputs
    // ================================================================================
    TF_Operation* output_op;
    output_op = TF_GraphOperationByName(graph, outputLayer.c_str());

    TF_Output* run_outputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
    run_outputs[0].oper = output_op;
    run_outputs[0].index = 0;


    TF_Tensor** run_output_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
    float* raw_output_data = (float*)malloc(inputHeight*inputWidth * sizeof(float));
    raw_output_data[0] = 1.f;
    int64_t* raw_output_dims = (int64_t*)malloc(4 * sizeof(int64_t));
    raw_output_dims[0] = 1;
    raw_output_dims[1] = inputHeight;
    raw_output_dims[2] = inputWidth;
    raw_output_dims[3] = 1;

    TF_Tensor* output_tensor = TF_NewTensor(TF_FLOAT,
                                          raw_output_dims, 4,
                                          raw_output_data, inputHeight*inputWidth * sizeof(float),
                                          deallocator,
                                          NULL
                                         );
    run_output_tensors[0] = output_tensor;

    // run network
    // ================================================================================
    TF_SessionRun(sess,
                /* RunOptions */         NULL,
                /* Input tensors */      run_inputs, run_inputs_tensors, 1,
                /* Output tensors */     run_outputs, run_output_tensors, 1,
                /* Target operations */  NULL, 0,
                /* RunMetadata */        NULL,
                /* Output status */      status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(status));
        return;
    }

    // printf("output-tensor has %i dims\n", TF_NumDims(run_output_tensors[0]));

    void* output_data = TF_TensorData(run_output_tensors[0]);
    RawTensor2Yuv(piSrc, (float *)output_data, inputHeight, inputWidth, iStride);

    free((void*) run_inputs);
    free((void*) run_outputs);
    free((void*) run_inputs_tensors);
    free((void*) run_output_tensors);
    free((void*) raw_input_dims);
    free((void*) raw_output_dims);
}

#endif