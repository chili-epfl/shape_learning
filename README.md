shape_learning is a library for simultaneously learning multiple shape models based on user input, such as demonstrations.

Python dependencies:
```
enum, scipy, recordtype
```

The `ShapeModeler` class manages a model of a particular shape, for example a letter. The current implementation uses a principle component analysis-based model: it takes a dataset of instances of the shape and determines the parameters which explain the majority of variance in the dataset. 

The `ShapeLearner` class manages the learning of the parameters of a particular `ShapeModeler` attribute. The parameters of user-demonstrated shapes may be determined from the model, and used to update the learned parameters of the system.

The `ShapeLearnerManager` class manages collections of multiple `ShapeLearner`s. For example, in the context of learning words, the `ShapeLearnerManager` keeps track of the current word being learnt and its associated `ShapeLearner`s, in addition to the information on each letter/word which has been previously seen. This allows for long-term system memory, even when the shapes are not always part of the active collection.

A sample application using of learning words, with letter instances from the UJI pen charaters 2 dataset, is provided in `scripts/learning_letters.py` (additional dependency of Kivy). To run it, execute:
```
python learning_letters.py 'case'
```
