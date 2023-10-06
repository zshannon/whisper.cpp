//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//
#import "whisper.h"
//
//#include <stdint.h>
//
//// Define an enum for ggml_type
//typedef enum {
//    GGML_TYPE_F16,
//    GGML_TYPE_FP32,
//    GGML_TYPE_FP16,
//    GGML_TYPE_QX
//} ggml_type;
//
//// Forward declare whisper_model and whisper_vocab as opaque pointers
////typedef struct whisper_model whisper_model;
//typedef struct whisper_vocab whisper_vocab;
//
//typedef enum {
//    GGML_TENSOR_TYPE_1D,
//    GGML_TENSOR_TYPE_2D,
//    GGML_TENSOR_TYPE_3D,
//    // Add other possible types...
//} ggml_tensor_type;
//
//// Define whisper_kv_cache as an incomplete type (opaque pointer)
//typedef struct whisper_kv_cache whisper_kv_cache;
//
//// Define whisper_mel as an incomplete type (opaque pointer)
//typedef struct whisper_mel whisper_mel;
//
//// Define whisper_decoder as an incomplete type (opaque pointer)
////typedef struct whisper_decoder whisper_decoder;
//
//// Define kv_buf as an incomplete type (opaque pointer)
//typedef struct kv_buf kv_buf;
//
//// Define ggml_tensor as an incomplete type (opaque pointer)
//typedef struct ggml_tensor ggml_tensor;
//
//// Define whisper_segment as an incomplete type (opaque pointer)
//typedef struct whisper_segment whisper_segment;
//
//// Define whisper_vocab as an incomplete type (opaque pointer)
//typedef struct whisper_vocab whisper_vocab;
//
//// Define whisper_allocr as an incomplete type (opaque pointer)
////typedef struct whisper_allocr whisper_allocr;
//
//// Define whisper_coreml_context as an incomplete type (opaque pointer)
//typedef struct whisper_coreml_context whisper_coreml_context;
//
//// Define ggml_metal_context as an incomplete type (opaque pointer)
//typedef struct ggml_metal_context ggml_metal_context;
//
//// Define whisper_openvino_context as an incomplete type (opaque pointer)
//typedef struct whisper_openvino_context whisper_openvino_context;
//
//// Define the whisper_state structure
//typedef struct whisper_state {
//    int64_t t_sample_us;
//    int64_t t_encode_us;
//    int64_t t_decode_us;
//    int64_t t_prompt_us;
//    int64_t t_mel_us;
//
//    int32_t n_sample;
//    int32_t n_encode;
//    int32_t n_decode;
//    int32_t n_prompt;
//    int32_t n_fail_p;
//    int32_t n_fail_h;
//
//    whisper_kv_cache* kv_cross;
//    whisper_mel* mel;
//
//    whisper_decoder decoders[WHISPER_MAX_DECODERS];
//
//    kv_buf* kv_swap_bufs;
//
//    uint8_t* work_buffer;
//
//    whisper_allocr alloc_conv;
//    whisper_allocr alloc_encode;
//    whisper_allocr alloc_cross;
//    whisper_allocr alloc_decode;
//
//    ggml_tensor* embd_conv;
//    ggml_tensor* embd_enc;
//
//    float* logits;
//
//    whisper_segment* result_all;
//    whisper_token* prompt_past;
//
//    double* logits_id;
//
//    // Opaque type, replace with an appropriate C type.
//    void* rng;
//
//    int lang_id;
//
//    char* path_model;
//
//#ifdef WHISPER_USE_COREML
//    whisper_coreml_context* ctx_coreml;
//#endif
//
//#ifdef GGML_USE_METAL
//    ggml_metal_context* ctx_metal;
//#endif
//
//#ifdef WHISPER_USE_OPENVINO
//    whisper_openvino_context* ctx_openvino;
//#endif
//
//    int64_t t_beg;
//    int64_t t_last;
//    whisper_token tid_last;
//    float* energy;
//    int32_t exp_n_audio_ctx;
//} whisper_state;
//
//// Define the whisper_context structure
//typedef struct whisper_context {
//    int64_t t_load_us;
//    int64_t t_start_us;
//
//    ggml_type wtype;
//    ggml_type itype;
//
//    whisper_model* model;
//    whisper_vocab* vocab;
//    whisper_state* state;
//
//    char* path_model; // C-style string for path_model
//} whisper_context;
