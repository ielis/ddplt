import io
import setuptools

from ddplt import VERSION

setuptools.setup(name="ddplt",
                 version=VERSION,
                 packages=["ddplt"],
                 long_description=io.open('README.md', encoding='utf-8').read(),
                 long_description_content_type='text/markdown',
                 license='GPLv3',
                 install_requires=['matplotlib>=3.1'
                                   'numpy>=1.16',
                                   'pandas>=0.23'],

                 author="Daniel Danis",
                 author_email="daniel.gordon.danis@gmail.com",
                 url="",
                 description="Useful utility functions for evaluation of ML",
                 keywords="plot machine learning evaluation metrics",
                 test_suite='tests')
