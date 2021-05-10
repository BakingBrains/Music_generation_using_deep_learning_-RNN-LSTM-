import json
import numpy as np
import music21 as m21
from preprocessing import SEQUENCE_LENGTH, MAP_PATH
import tensorflow.keras as keras


class MusicGenerator:
    def __init__(self, model_path="Log/MusicGen.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAP_PATH, "r") as fp:
            self.mappings = json.load(fp)

        self.start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_music(self, seed, num_steps, max_seq_len, temperature):
        # seed is a encoded music string
        # creating seed with start symbol
        seed = seed.split()
        music = seed
        seed = self.start_symbols + seed
        # seed to int mapping
        seed = [self.mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # seed limit to max_seq_length
            seed = seed[-max_seq_len:]
            # one-hot encode
            seed_encode = keras.utils.to_categorical(seed, num_classes=len(self.mappings))
            # predict dimension = (1, max_seq_length, number of symbol in vocabulary)
            seed_encode = seed_encode[np.newaxis, ...]

            # prediction
            probabilities = self.model.predict(seed_encode)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            # update seed
            seed.append(output_int)
            # int to encoding mapping
            out_symbol = [k for k, v in self.mappings.items() if v == output_int][0]
            # check for end of music
            if out_symbol == "/":
                break

            # else update music
            music.append(out_symbol)

        return music

    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_music(self, music, step_duration=0.25, format="midi", file_name="music.mid"):
        # create a music21 stream
        stream = m21.stream.Stream()
        # parse all symbols to create note/rest obj
        start_symbol = None
        step_counter = 1

        for i , symbol in enumerate(music):
            # handle case for note/rest
            if symbol != "_" or i + 1 == len(music):
                # dealing with note/rest after first ome
                if start_symbol is not None:
                    quarter_len_duration = step_duration * step_counter
                    # for rest encountered
                    if start_symbol == "r":
                        event_m21 = m21.note.Rest(quarterLength=quarter_len_duration)
                    # for note encountered
                    else:
                        event_m21 = m21.note.Note(int(start_symbol), quarterLength=quarter_len_duration)

                    stream.append(event_m21)

                    # reset step_counter
                    step_counter = 1
                start_symbol = symbol
            # handle case for prolongation "_"
            else:
                step_counter += 1
        # stream to midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MusicGenerator()
    seed = "55 _ 60 _ 60 _ 60 _ 62 _ 64"
    music = mg.generate_music(seed, 500, SEQUENCE_LENGTH, 0.9)
    print(music)
    mg.save_music(music)