from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn            
import nvidia.dali.types as types
# import matplotlib.pyplot as plt
import numpy as np
from nvidia.dali.pipeline import Pipeline

batch_size = 1
audio_files = "/root/audio"

@pipeline_def
def audio_decoder_pipe():
    encoded, _ = fn.readers.file(name="Reader", pad_last_batch=False, device="cpu", file_root=audio_files, shard_id=0, num_shards=1, shuffle_after_epoch=True)
    audio, sr = fn.decoders.audio(encoded, dtype=types.FLOAT, sample_rate=1600, downmix=True, device="cpu")
    return audio, sr

pipe = audio_decoder_pipe(batch_size=batch_size, num_threads=4, device_id=None)
pipe.build()          
cpu_output = pipe.run()
print(cpu_output)


audio_data = cpu_output[0].at(0)
sampling_rate = cpu_output[1].at(0)
print("Sampling rate:", sampling_rate, "[Hz]")
print("Audio data:", audio_data)
audio_data = audio_data.flatten()
print("Audio data flattened:", audio_data)

# batch_size = 2
# test_data_shape = [10, 20, 3]

# def test_move_to_device_end():
#     test_data_shape = [1, 3, 0, 4]
#     def get_data():
#         out = [np.empty(test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
#         return out

#     pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
#     outs = fn.external_source(source=get_data)
#     pipe.set_outputs(outs)
#     pipe.build()

# def test_audio_decoder_cpu():
#     pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
#     input, _ = fn.readers.file(files=audio_files, shard_id=0, num_shards=1)
#     decoded, _ = fn.decoders.audio(input)
#     pipe.set_outputs(decoded)
#     pipe.build()
#     for _ in range(3):
#         pipe.run()

# test_move_to_device_end()
# test_audio_decoder_cpu()


# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
# def test():
#     pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
#     audio, label = ops.FileReader(name="Reader", pad_last_batch=(pipeline_type == 'val'), device="cpu", file_root=file_root, file_list=sampler.get_file_list_path(), shard_id=shard_id,
#                                    num_shards=n_shards, shuffle_after_epoch=shuffle)()
