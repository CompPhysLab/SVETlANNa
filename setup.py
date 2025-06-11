from distutils.core import setup
setup(
    name = 'svetlanna',
    packages = ['svetlanna'],
    version = '1.0.0',
    license='MPL-2.0',
    description = 'SVETlANNa is an open-source Python library for simulation of free-space optical set-ups and neuromorphic systems such as Diffractive Neural Networks.',   # Give a short description about your library
    author = ['Vladimir Igoshin', 'Semen Chugunov', 'Denis Sakhno', 'Alexey Kokhanovskyi', 'Alexey Shcherbakov'],
    author_email = 'cplab@metalab.ifmo.ru',
    url = 'https://github.com/CompPhysLab/SVETlANNa',
    download_url = 'https://github.com/CompPhysLab/SVETlANNa/archive/refs/tags/v1.0.0.tar.gz',
    keywords = ['OPTICAL NEURAL NETWORK', 'DIFFRACTIVE NEURAL NETWORK', 'OPTICAL BEAM', 'OPTICAL SETUP', 'DIFFRACTIVE ELEMENT', 'SPATIAL LIGHT MODULATOR'],   # Keywords that define your package best
    install_requires=[
          'typing',
          'types',
          'torch',
          'functools',
          'pathlib',
          'json',
          're',
          'datetime',
          'contextlib',
          'enum',
          'io',
          'warnings',
          'numpy',
          'abc',
          'collections',
          'dataclasses',
          'PIL',
          'numbers',
          'base64',
          'anywidget',
          'traitlets',
          'jinja2',
    ],
    classifiers=[
        'Development Status :: 5 - Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers, Engineers, Opticians',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MPL-2.0',   # Again, pick a license
        'Programming Language :: Python :: 3.11',      #Specify which pyhton versions that you want to support
    ],
)