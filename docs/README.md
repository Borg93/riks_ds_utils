# riks_ds_utils

Long description of project.

## Package versioning

Package version
PEP0440 compatible formatted version, see:
https://www.python.org/dev/peps/pep-0440/

Generic release markers:
X.Y.0 # For first release after an increment in Y
X.Y.Z # For bugfix releases

Admissible pre-release markers:
X.Y.ZaN # Alpha release
X.Y.ZbN # Beta release
X.Y.ZrcN # Release Candidate
X.Y.Z # Final release

Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
'X.Y.dev0' is the canonical version of 'X.Y.dev'

For instance:
X increments -> Incompatible w/ previous versions and has foundational changes
Y incremnts -> Adding functionality, but still backwards compatible
Z increments -> Backwards compatible and focuses on bug fixes (i.e Defects that are in production)

## When and what to test?

- When you change code, add a test
- Test the outcome, not the implementation
- When you find a bug, add a test
- Help identify complexity
- Dont test code that is already tested!

However in DS (when problem is not determinstic..), it could be better to test

- Test properties, not specific values
- Assumptions about the data type and shape
- Test probabalistically

good packages:
pytest with fixtures, parametrize and mocks..
https://hypothesis.readthedocs.io/en/latest/ ideal for code that accepts code "from the wild /dirty data"
