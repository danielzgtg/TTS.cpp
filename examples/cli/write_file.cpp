#include <iostream>
#include "write_file.h"
#include "audio_file.h"

void write_audio_file(std::string path, const tts_response & data) {
    if (path.empty()) {
        path = "TTS.cpp.wav";
    }
    std::cout << "Writing audio file: " << path << std::endl;

    AudioFile<float> file;
    file.samples[0] = std::vector(data.data, data.data + data.n_outputs);
    file.save(path, AudioFileFormat::Wave);
    file.printSummary();
}
