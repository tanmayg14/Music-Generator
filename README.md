# Music-Generator
Prerequisites:-

Libraries:
Numpy
Pickle
Tensorflow
Music21

Execution:
1. Run the code model_generate.ipynb on google colab on GPU Hardware Accelerator for the most efficient and fast training on models. Then the weights of your trained model will be downloaded. 
2. Then run the generator.py or generator.ipynb file, they serve the same purpose.
3. Then input the required 20 nodes or chords for the first 10 seconds. The interval of each node and chord is 0.5 seconds. The mid file is then downloaded and you can listen to your generated song.
4. If you don't want to generate your own music but want any random music then run auto_generate.py .
5. Now the generated mid file is ready to run. So run the listen.ipynb file and can listen to the generated music.

Note:-
	Everytime you generate a mid file then the previous mid file will get replaced.

Variations
1. You can also change the number of iterations of the training model.
2. If you want your generated music to be similar to some other songs. Then the training dataset can be changed and different mid files can be uploaded in the training data folder. 
3. You can also change the model and try different layers.
4. You can also change the length of the initial tone by changing the sequence length to twice the duration required. 
5. You can also increase the duration of the generated music by changing the existing range of 200 to any required value.
