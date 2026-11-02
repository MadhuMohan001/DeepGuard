# Sample notes (replace with your IEEE paper and project report)

DeepGuard classifies a video as real or fake by fusing visual features from ResNet-50
with audio features from Wav2Vec 2.0 using a cross-attention layer. Cross-attention lets
the visual stream attend to the audio stream, so mismatches between lip movement and
speech can be detected.

A confidence score close to 0.5 means the model is uncertain, and such videos should be
reviewed manually. The model was evaluated on the FakeAVCeleb dataset and reached 97.33%
accuracy. Performance can drop on manipulation methods or video quality not seen in training.
