from music21 import converter, instrument, note, chord, stream
import glob
import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model

def generate(start):
	model = load_model('model.hdf5')
	notes=[]
	with open("notes", 'rb') as f:
	    notes= pickle.load(f)

	pitchnames = sorted(set(notes))
	ele_to_int = dict( (ele, num) for num, ele in enumerate(pitchnames) )
	n_vocab = len(set(notes))

	sequence_length = 20
	network_input = []

	for i in range(len(notes) - sequence_length):
	    seq_in = notes[i : i+sequence_length] # contains 100 values
	    network_input.append([ele_to_int[ch] for ch in seq_in])

	int_to_ele = dict((num, ele) for num, ele in enumerate(pitchnames))

	pattern = []
	prediction_output = []

	for i in start:
		pattern.append(i)
		prediction_output.append(int_to_ele[i])

	for note_index in range(200):
	    prediction_input = np.reshape(pattern, (1, len(pattern), 1)) # convert into numpy desired shape 
	    prediction_input = prediction_input/float(n_vocab)
	    
	    prediction =  model.predict(prediction_input, verbose=0)
	    
	    idx = np.argmax(prediction)
	    result = int_to_ele[idx]
	    prediction_output.append(result) 
	    
	    # Remove the first value, and append the recent value.. 
	    # This way input is moving forward step-by-step with time..
	    pattern.append(idx)
	    pattern = pattern[1:]

	offset = 0 # Time
	output_notes = []

	for patterns in prediction_output:
	    
	    # if the pattern is a chord
	    if ('+' in patterns) or patterns.isdigit():
	        notes_in_chord = patterns.split('+')
	        temp_notes = []
	        for current_note in notes_in_chord:
	            new_note = note.Note(int(current_note))  # create Note object for each note in the chord
	            new_note.storedInstrument = instrument.Piano()
	            temp_notes.append(new_note)
	            
	        
	        new_chord = chord.Chord(temp_notes) # creates the chord() from the list of notes
	        new_chord.offset = offset
	        output_notes.append(new_chord)
	    
	    else:
	            # if the pattern is a note
	        new_note = note.Note(patterns)
	        new_note.offset = offset
	        new_note.storedInstrument = instrument.Piano()
	        output_notes.append(new_note)
	        
	    offset += 0.5
	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp='generated_output.mid')
	return 

def show():
	notes=[]
	with open("notes", 'rb') as f:
	    notes= pickle.load(f)

	pitchnames = sorted(set(notes))
	ele_to_int = dict( (ele, num) for num, ele in enumerate(pitchnames) )
	for i in ele_to_int:
		print(ele_to_int[i],":",i)
	return

if __name__== "__main__":
	show()
	start=[]
	for i in range(20):
		x=int(input())
		if x>358:
			print("This note or chord doesnt exist")
			break
		start.append(i)
	if len(start)==20:
		generate(start)