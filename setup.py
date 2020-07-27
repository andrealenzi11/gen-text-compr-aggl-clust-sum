from distutils.core import setup

# read the contents of your README file
from os import path
content_root_directory = path.abspath(path.dirname(__file__))
with open(path.join(content_root_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

VERSION = '0.0.3'

setup(
    name='gtcacs',  # How you named your package folder (MyLib)
    packages=['gtcacs'],  # Chose the same as "name"
    version=VERSION,
    license='MIT',
    description='Generative Text Compression with Agglomerative Clustering Summarization (GTCACS)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Andrea Lenzi',
    author_email='andrealenzi11@gmail.com',
    url='https://github.com/andrealenzi11/gen-text-compr-aggl-clust-sum.git',
    download_url=f'https://github.com/andrealenzi11/gen-text-compr-aggl-clust-sum/archive/{VERSION}.tar.gz',
    keywords=[
        'discussion topics',
        'topic modeling',
        'topic modelling',
        'topic extraction',
    ],
    install_requires=[
        'tensorflow==2.2.0',
        'scikit-learn==0.23.1',
        'numpy==1.19.1',
        'scipy==1.4.1',
        'tqdm==4.48.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
