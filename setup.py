import setuptools
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

python_versions = '>=3.7, <3.11'  # not tested with any other than 3.10, but probably works fine

# conda create -n rm4d python=3.10
# conda activate rm4d
# conda install -c conda-forge pybullet numpy scipy matplotlib tqdm
requirements_default = [
    'numpy',
    'pybullet',
    'scipy',
    'matplotlib',
    'tqdm',
]

# add files to package data
all_files = []
data_dir = Path('rm4d/robots/assets/')
for p in data_dir.rglob('*'):
    if p.is_file():
        # remove rm4d/ and convert to string
        all_files.append(str(Path(*p.parts[1:])))

package_data = {
    '': all_files
}

setuptools.setup(
    name='rm4d',
    version='0.1.0',
    python_requires=python_versions,
    install_requires=requirements_default,
    packages=setuptools.find_packages(),
    zip_safe=False,
    package_data=package_data,
    url='https://github.com/mrudorfer/rm4d',
    license='MIT License',
    author='Martin Rudorfer',
    author_email='m.rudorfer@aston.ac.uk',
    description='Reachability Map 4D',
    long_description=long_description
)
