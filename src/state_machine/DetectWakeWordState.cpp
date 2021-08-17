#include "I2SSampler.h"
#include "AudioProcessor.h"
#include "NeuralNetwork.h"
#include "RingBuffer.h"
#include "DetectWakeWordState.h"
#include "esp_log.h"
#include <sys/time.h>

#define WINDOW_SIZE 256
#define STEP_SIZE 128
#define POOLING_SIZE 3
#define AUDIO_LENGTH 16000

static const char *TAG = "APP";

DetectWakeWordState::DetectWakeWordState(I2SSampler *sample_provider)
{
    // save the sample provider for use later
    m_sample_provider = sample_provider;
    // some stats on performance
    m_average_detect_time = 0;
    m_number_of_runs = 0;
}
void DetectWakeWordState::enterState()
{
    // Create our neural network
    m_nn = new NeuralNetwork();
    ESP_LOGI(TAG, "Created Neural Net");
    // create our audio processor
    m_audio_processor = new AudioProcessor(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
    ESP_LOGI(TAG, "Created audio processor");

    m_number_of_detections = 0;
}
bool DetectWakeWordState::run()
{
    // time how long this takes for stats
    struct timeval tv_now;
    gettimeofday(&tv_now, NULL);
    int64_t start = (int64_t)tv_now.tv_sec * 1000L + (int64_t)tv_now.tv_usec / 1000L;
    // get access to the samples that have been read in
    RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
    // rewind by 1 second
    reader->rewind(16000);
    // get hold of the input buffer for the neural network so we can feed it data
    float *input_buffer = m_nn->getInputBuffer();
    // process the samples to get the spectrogram
    m_audio_processor->get_spectrogram(reader, input_buffer);
    // finished with the sample reader
    delete reader;
    // get the prediction for the spectrogram
    float output = m_nn->predict();
    gettimeofday(&tv_now, NULL);
    int64_t end = (int64_t)tv_now.tv_sec * 1000L + (int64_t)tv_now.tv_usec / 1000L;   
    // compute the stats
    m_average_detect_time = (end - start) * 0.1 + m_average_detect_time * 0.9;
    m_number_of_runs++;
    // log out some timing info
    if (m_number_of_runs == 100)
    {
        m_number_of_runs = 0;
        ESP_LOGI(TAG, "Average detection time %.fms\n", m_average_detect_time);
    }

    // log out the output
    ESP_LOGI(TAG, "Neural network output is: %f \n", output);

    // use quite a high threshold to prevent false positives
    if (output > 0.95)
    {
        m_number_of_detections++;
        if (m_number_of_detections > 1)
        {
            m_number_of_detections = 0;
            // detected the wake word in several runs, move to the next state
            ESP_LOGI(TAG, "P(%.2f): Here I am, brain the size of a planet...\n", output);
            return true;
        }
    }
    // nothing detected stay in the current state
    return false;
}
void DetectWakeWordState::exitState()
{
    // Create our neural network
    delete m_nn;
    m_nn = NULL;
    delete m_audio_processor;
    m_audio_processor = NULL;
    uint32_t free_ram = esp_get_free_heap_size();
    ESP_LOGI(TAG, "Free ram after DetectWakeWord cleanup %d\n", free_ram);
}