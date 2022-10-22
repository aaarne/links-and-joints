import setuptools

setuptools.setup(
    name="links-and-joints",
    version="0.0.1",
    author="Arne Sachtler",
    author_email="arne.sachtler@tum.de",
    description="Dynamical Models of Planar Mechanical Systems",
    url="https://github.com/aaarne/links-and-joints",
    packages=['links_and_joints', 'links_and_joints.planarrobots'],
    install_requires=['sympy'],
)