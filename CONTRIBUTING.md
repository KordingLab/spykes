# Guidelines

- Check the [Gitter](https://gitter.im/KordingLab/spykes) to discuss the issue
- Please follow [Google's Python style guide](https://google.github.io/styleguide/pyguide.html), mostly especially for docstrings. You will probably notice if you've made mistakes with docstrings because it won't render correctly when you build the documentation.

# Testing

To run testing and build documentation, install the development dependencies:

```bash
pip install -e .[develop]
```

Make sure that tests are passing for both Python 2.7 and Python 3.6:

```bash
python setup.py test  # Unit tests
python setup.py flake  # Linting
```

Making sure these steps pass will help the continuous integration step go smoothly.

# Building Documentation

To build the documentation locally, run

```bash
cd doc/
make html
```
