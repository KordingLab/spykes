# Guidelines

- Check the [Gitter](https://gitter.im/KordingLab/spykes) to discuss the issue
- Please follow [Google's Python style guide](https://google.github.io/styleguide/pyguide.html), mostly

# Testing

Install the testing dependencies:

```bash
pip install flake8 nose
```

Make sure that tests are passing for both Python 2.7 and Python 3.6:

```bash
python setup.py test  # Unit tests
python setup.py flake  # Linting
```

Making sure these steps pass will help the continuous integration step go smoothly.
