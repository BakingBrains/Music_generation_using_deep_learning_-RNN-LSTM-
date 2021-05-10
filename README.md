# Music_generation_using_deep_learning_-RNN-LSTM-
Generating music using LSTM with the help of ESAC folk song dataset</br>

<b>Dataset</b>: https://kern.humdrum.org/cgi-bin/browse?l=essen%2Feuropa%2Fdeutschl </br>

<b>Libraries</b>:</br>
- music21 (pip install music21)</br>
- tensorflow</br>

<b>Software required</b>:</br>
- musescore: https://musescore.org/en/download</br>

After installing musescore, run the music21Environment.py with the correct path to musescore</br>

* Download and copy the dataset to <i>'Dataset'</i> folder </br>
* The above provided file <i>'file_dataset'</i> is the generated file from the preprocessing part for folder named <i>'erk'</i> in Dataset folder</br>
* <i>mapping.json</i>e is the json file for number of keys in the music file inside erk folder</br>
* The trained model will be saved in log folder and used in musicgenerator.py file</br>

<b>File running sequence</b>:</br>
- music21Environment.py
- preprocessing.py
- train.py
- musicgenerator.py
