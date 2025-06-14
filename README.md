# Data Science and AI for Medicine Training School 2025

Resources and materials for a DataScience / AI training school for the medical domain, conducted as part of the Come2Data (https://come2data.de/) project.

You can read the training content under this URL:

[https://scads.github.io/ai4medicine-2025](https://scads.github.io/ai4medicine-2025)

The following instructions are just for contributors and trainers.

## Development

### Building the book

If you'd like to develop and/or build the book, you should:

1. Clone this repository
2. Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
3. (Optional) Edit the books source files.
4. Run `jupyter-book clean .` to remove any existing builds
5. Run `jupyter-book build .`

A fully-rendered HTML version of the book will be built in `/_build/html/`. Make sure to never upload this folder to github.

### Hosting the book

Please see the [Jupyter Book documentation](https://jupyterbook.org/publish/web.html) to discover options for deploying a book online using services such as GitHub, GitLab, or Netlify.

For GitHub and GitLab deployment specifically, the [cookiecutter-jupyter-book](https://github.com/executablebooks/cookiecutter-jupyter-book) includes templates for, and information about, optional continuous integration (CI) workflow files to help easily and automatically deploy books online with GitHub or GitLab. For example, if you chose `github` for the `include_ci` cookiecutter option, your book template was created with a GitHub actions workflow file that, once pushed to GitHub, automatically renders and pushes your book to the `gh-pages` branch of your repo and hosts it on GitHub Pages when a push or pull request is made to the main branch.

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/ScaDS/ai4medicine/graphs/contributors).

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).

We acknowledge the financial support by the Federal Ministry of Education and Research of Germany and by Sächsische Staatsministerium für Wissenschaft, Kultur und Tourismus in the programmes „Stärkung der Datenkompetenzen des wissenschaftlichen Nachwuchses“, project identification number: 16DKZ2044; and Center of Excellence for AI-research „Center for Scalable Data Analytics and Artificial Intelligence Dresden/Leipzig“, project identification number: ScaDS.AI.
