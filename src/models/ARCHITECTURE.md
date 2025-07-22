# File layout

```mermaid
erDiagram
    context }o--|| model
```

For the average model:
1. src/models/loaders.cpp:runner_from_file
   1. model_name/loader.cpp:model_name_from_file
      1. model_name/hparams.cpp
      2. src/decoder/decoder_model_name/context.cpp:build_new_decoder_model_name_context
      3. model_name/context.cpp:build_new_model_name_context
      4. model_name/tensors.cpp:assign_weight
         1. model_name/subunit/tensors.cpp
         2. src/decoder/decoder_model_name/tensors.cpp:assign_weight
            1. ...
      5. model_name/build.cpp:prepare_post_load
         1. src/decoder/decoder_model_name/build.cpp:prepare_post_load
            1. ...
         2. model_name/kv.cpp
         3. model_name/build.cpp:build_model_name_graph
             1. model_name/subunit/build.cpp
   2. src/tts.cpp:generate
      1. model_name/context.cpp:run
         1. model_name/tokenize.cpp
         2. model_name/build.cpp:build_model_name_graph
            1. ...
         3. src/decoder/decoder_model_name/context.cpp:run
            1. src/decoder/decoder_model_name/build.cpp
               1. ...

build, hparams, (model_)loader, (load_)tensors, etc. mean what they do in llama.cpp.
