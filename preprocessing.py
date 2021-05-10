import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

DATASET_PATH = "Dataset/erk"
ACCEPTABLE_DURATIONS = [ 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4 ]
SAVE_DIR = "dataset_out" # create if doesn't exist
SINGLE_FILE_DATASET = "file_dataset"
MAP_PATH = "mapping.json"
SEQUENCE_LENGTH = 64


def load_song(dataset_path):
    songs = []

    # ran through the file and load
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs


def acceptable_time(song, acceptable_dur):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_dur:
            return False
    return True


def transpose(song):
    # get key from song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # estimate key with music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    #print(key)
    # Interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    # transpose song by obtained interval
    transposed_song = song.transpose(interval)

    return transposed_song


def encode(song, time_step = 0.25):
    encoded_song = []

    for event in song.flat.notesAndRests:
        # notes handling
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi #picth
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        # convert the note/rest into time series
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # convert encoded song to string
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):
    pass

    # load
    print("Loading songs...")
    songs = load_song(dataset_path)
    print(f"Loaded {len(songs)} songs")

    for i, song in enumerate(songs):
        # filter data
        if not acceptable_time(song, ACCEPTABLE_DURATIONS):
            continue
        # Transpose songs to Amin/Cmaj
        song = transpose(song)
        # encode songs with time series representation
        encoded_song = encode(song)
        # save song to .txt
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_dataset_file(datasetout_path, file_dataset_path, sequence_length):
    # load encoded songs and adding delimiters
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    for path, _, files in os.walk(datasetout_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]
    # save the string which contain all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_dataset_mapping(songs, map_path):
    # identify vocabulary
    mappings = {}
    songs = songs.split()
    vocab = list(set(songs))

    # mapping
    for i, symbol in enumerate(vocab):
        mappings[symbol] = i
    # save it to json file
    with open(map_path, "w") as fp:
        json.dump(mappings, fp, indent=2)


def songs_to_int(songs):
    # load mappings
    int_songs = []
    with open(MAP_PATH, "r") as fp:
        mappings = json.load(fp)
    # cast song string to list
    songs = songs.split()
    # map to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequence(sequence_length):
    # [1,2,3,4,...] -> i: [1,2], t:3; i:[2, 3], t:4;...... sliding one by one

    # load songs and map it to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = songs_to_int(songs)
    # generate training sequence
    # 100 symbols, 64 sequence length then training sequence =100-64
    inputs = []
    targets =[]
    num_seq = len(int_songs) - sequence_length
    for i in range(num_seq):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    # one-hot encode
    vocab_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocab_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(DATASET_PATH)
    songs =  create_dataset_file(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_dataset_mapping(songs, MAP_PATH)
#    inputs, targets = generate_training_sequence(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()


#    songs = load_song(DATASET_PATH)
#    print(f"Loaded {len(songs)} songs")
#    song = songs[0]
#    print(f"Duration Accepted? {acceptable_time(song, ACCEPTABLE_DURATIONS)}")
#    transpose_song = transpose(song)
#    song.show()
#   transpose_song.show()