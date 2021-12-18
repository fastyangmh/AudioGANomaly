#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict_gui import BasePredictGUI
from src.predict import Predict
from DeepLearningTemplate.data_preparation import AudioLoader, parse_transforms
from tkinter import Button, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from playsound import playsound
import tkinter as tk


# class
class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=('.wav'))
        self.predictor = Predict(project_parameters=project_parameters)
        self.classes = project_parameters.classes
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.sample_rate = project_parameters.sample_rate
        assert project_parameters.threshold is not None, 'please check the threshold. the threshold value is {}.'.format(
            project_parameters.threshold)
        self.threshold = project_parameters.threshold

        # button
        self.play_button = Button(master=self.window,
                                  text='Play',
                                  command=self.play)

        # matplotlib canvas
        # this is Tkinter default background-color
        facecolor = (0.9254760742, 0.9254760742, 0.9254760742)
        figsize = np.array([12, 4]) * project_parameters.in_chans
        self.image_canvas = FigureCanvasTkAgg(Figure(figsize=figsize,
                                                     facecolor=facecolor),
                                              master=self.window)

    def reset_widget(self):
        super().reset_widget()
        self.image_canvas.figure.clear()

    def display(self):
        waveform = self.loader(path=self.filepath)
        # the transformed sample dimension is (in_chans, freq, time)
        sample = self.transform(waveform)
        sample = sample.cpu().data.numpy()
        # invert the freq axis so that the frequency axis of the spectrogram is displayed correctly
        sample = sample[:, ::-1, :]
        rows, cols = len(sample), 2
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            if idx % cols == 1:
                # plot waveform
                subplot.title.set_text(
                    'channel {} waveform'.format((idx - 1) // cols + 1))
                subplot.set_xlabel('time')
                subplot.set_ylabel('amplitude')
                time = np.linspace(
                    0, len(waveform[(idx - 1) // cols]),
                    len(waveform[(idx - 1) // cols])) / self.sample_rate
                subplot.plot(time, waveform[(idx - 1) // cols])
            else:
                # plot spectrogram
                # TODO: display frequency and time.
                subplot.title.set_text(
                    'channel {} spectrogram'.format((idx - 1) // cols + 1))
                subplot.imshow(sample[(idx - 1) // cols])
                subplot.axis('off')
        self.image_canvas.draw()

    def open_file(self):
        super().open_file()
        self.display()

    def display_output(self, fake_sample):
        self.image_canvas.figure.clear()
        waveform = self.loader(path=self.filepath)
        # the transformed sample dimension is (in_chans, freq, time)
        sample = self.transform(waveform)
        # convert the range of the sample to 0~1
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        sample = sample.cpu().data.numpy()
        # invert the freq axis so that the frequency axis of the spectrogram is displayed correctly
        sample = sample[:, ::-1, :]
        # the fake_sample dimension is (1, in_chans, freq, time),
        # so use 0 index to get the first fake_sample
        fake_sample = fake_sample[0][:, ::-1, :]
        diff = np.abs(sample - fake_sample)
        rows, cols = len(sample), 3
        title = ['real', 'fake', 'diff']
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            subplot.title.set_text('{} {}'.format(title[(idx - 1) % 3],
                                                  ((idx - 1) // cols + 1)))
            if (idx - 1) % 3 == 0:
                # plot real
                subplot.imshow(sample[(idx - 1) // cols])
            elif (idx - 1) % 3 == 1:
                # plot fake
                subplot.imshow(fake_sample[(idx - 1) // cols])
            elif (idx - 1) % 3 == 2:
                # plot diff
                subplot.imshow(diff[(idx - 1) // cols])
            subplot.axis('off')
        self.image_canvas.draw()

    def recognize(self):
        if self.filepath is not None:
            score, fake_sample = self.predictor.predict(filepath=self.filepath)
            self.display_output(fake_sample=fake_sample)
            score = score.item()  # score is a scalar
            self.predicted_label.config(text='score:\n{}'.format(score))
            self.result_label.config(text=self.classes[int(
                score >= self.threshold)])
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def play(self):
        if self.filepath is not None:
            playsound(sound=self.filepath, block=True)
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def run(self):
        # NW
        self.open_file_button.pack(anchor=tk.NW)
        self.recognize_button.pack(anchor=tk.NW)
        self.play_button.pack(anchor=tk.NW)

        # N
        self.filepath_label.pack(anchor=tk.N)
        self.image_canvas.get_tk_widget().pack(anchor=tk.N)
        self.predicted_label.pack(anchor=tk.N)
        self.result_label.pack(anchor=tk.N)

        # run
        super().run()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # launch prediction gui
    PredictGUI(project_parameters=project_parameters).run()
