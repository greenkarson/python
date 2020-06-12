import torchaudio
import matplotlib.pyplot as plt

waveform, sample_rate = torchaudio.load(r"E:\dataset\audio\SpeechCommands\speech_commands_v0.02\bed\00f0204f_nohash_0.wav")

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())

# specgram = torchaudio.transforms.Spectrogram()(waveform)
#
# print("Shape of spectrogram: {}".format(specgram.size()))
#
# plt.figure()
# plt.imshow(specgram.log2()[0, :, :].numpy(), cmap='gray')

specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')

plt.show()
