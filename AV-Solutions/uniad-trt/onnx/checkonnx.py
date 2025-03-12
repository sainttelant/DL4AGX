import tensorrt as trt
logger = trt.Logger(trt.Logger.VERBOSE)

build = trt.Builder(logger)

network = build.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)