# Deep Reinforcement Learning In Action

Code Snippets from the [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action) book from Manning, Inc

## How this is Organized

The code snippets, listings, and projects are all embedded in Jupyter Notebooks
organized by chapter. Visit [http://jupyter.org/install](http://jupyter.org/install) for
instructions on installing Jupyter Notebooks.

The chapter directories contain the maintained notebooks. See [Book compatibility and corrections](BOOK_COMPATIBILITY.md) for API updates and corrections to printed listings, and [Errata](Errata/README.md) for links to maintained chapters and the immutable pre-refresh source. Model architectures and the methods described in the text are preserved; stochastic learning results vary between runs.

## Requirements

In order to run many of the projects, you'll need at least the [NumPy](http://www.numpy.org/) library
and [PyTorch](http://pytorch.org/).

```
python -m pip install -r requirements.txt
```

Use a fresh Python 3.13 environment and launch each notebook from its chapter directory. Run `python -m unittest discover -s tests -v` for the fast checks. See [validation results](VALIDATION.md) for full training outcomes and tested limits.

## Contribute

If you experience any issues running the examples, please file an issue.
If you see typos or other errors in the book, please edit the [Errata.md](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Errata.md) file and create a pull request.
