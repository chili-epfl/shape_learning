shape_learning is a library for simultaneously learning multiple shape models based on user input, such as demonstrations.

Python dependencies:
```
enum, scipy, recordtype
```

The `ShapeModeler` class manages a model of a particular shape, for example a letter. The current implementation uses a principle component analysis-based model: it takes a dataset of instances of the shape and determines the parameters which explain the majority of variance in the dataset. 

A GUI to visualise the effect of varying parameters of the shape model is provided in `scripts/shape_model_gui.py`.  Sample usage:
```
python shape_model_gui.py 'd' 5
```
![GUI screenshot of letter 'd'](https://github.com/dhood/shape_learning/raw/master/doc/gui_d_params1.png)
*An example of the GUI for visualising the effects of the model of 'd'.*

The `ShapeLearner` class manages the learning of the parameters of a particular `ShapeModeler` attribute. The parameters of user-demonstrated shapes may be determined from the model, and used to update the learned parameters of the system.

The `ShapeLearnerManager` class manages collections of multiple `ShapeLearner`s. For example, in the context of learning words, the `ShapeLearnerManager` keeps track of the current word being learnt and its associated `ShapeLearner`s, in addition to the information on each letter/word which has been previously seen. This allows for long-term system memory, even when the shapes are not always part of the active collection.

A sample application of learning words is provided in `scripts/learning_letters.py` (additional dependency of Kivy). The default dataset of letter instances is from the UJI pen charaters 2 dataset. Sample usage:
```
python learning_letters.py 'case'
```
![Letter learning app screenshot](https://github.com/dhood/shape_learning/raw/master/doc/learning_a_demo.png)
*An example of the app for demonstrating the letter 'a' (left) to update the system-learned shape (right, originally 'o'-shaped).*

For a more complex use case of the shape_learning library, see [the CoWriter project](https://github.com/chili-epfl/cowriter_letter_learning).
