import io
import os
import re

from setuptools import find_packages
from setuptools import setup

REQUIRED=['ipywidgets>=7.2.1', 'matplotlib>=3.1.1', 'numpy>=1.16.4', 'mlxtend>=0.17.1', 'pandas>=0.24.2','plotly>=4.5.0', 'python_igraph>=0.8.3', 'scikit_learn>=0.23.2']


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="divexplorer",
    version="0.1.1",
    url="https://github.com/elianap/divexplorer.git",
    license='MIT',

    author="Eliana Pastor",
    author_email="eliana.pastor@polito.it",

    description="DivExplorer",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",

    packages=find_packages(exclude=('tests','notebooks')),

    install_requires=REQUIRED,

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
