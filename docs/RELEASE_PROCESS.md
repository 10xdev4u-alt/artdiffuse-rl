# Release Process

## Overview

This document describes the process for releasing new versions of ArtDiffuse-RL.

## Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Release Checklist

### Pre-Release

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated
- [ ] Version number is updated
- [ ] Release notes are written

### Release

- [ ] Create a release branch
- [ ] Tag the release
- [ ] Build and upload packages
- [ ] Create GitHub release
- [ ] Announce the release

### Post-Release

- [ ] Merge release branch to main
- [ ] Update documentation
- [ ] Close related issues
- [ ] Plan next release

## Release Steps

### 1. Prepare Release

```bash
# Create release branch
git checkout -b release/v0.2.0

# Update version
# Edit src/__init__.py and setup.py

# Update CHANGELOG.md
# Add release notes for v0.2.0

# Commit changes
git add -A
git commit -m "chore: prepare release v0.2.0"
```

### 2. Test Release

```bash
# Run tests
pytest tests/ -v

# Test installation
pip install -e .

# Test import
python -c "import artdiffuse_rl; print(artdiffuse_rl.__version__)"
```

### 3. Create Release

```bash
# Tag release
git tag -a v0.2.0 -m "Release v0.2.0"

# Push tag
git push origin v0.2.0

# Create GitHub release
gh release create v0.2.0 --title "v0.2.0" --notes-file RELEASE_NOTES.md
```

### 4. Build and Upload

```bash
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

### 5. Announce Release

- [ ] Post on GitHub
- [ ] Post on social media
- [ ] Send newsletter (if applicable)
- [ ] Update website (if applicable)

## Release Notes Template

```markdown
# Release v0.2.0

## What's New

- Feature 1
- Feature 2
- Feature 3

## Bug Fixes

- Fix 1
- Fix 2

## Breaking Changes

- Change 1
- Change 2

## Contributors

- @contributor1
- @contributor2

## Full Changelog

[Compare with previous release](https://github.com/10xdev4u-alt/artdiffuse-rl/compare/v0.1.0...v0.2.0)
```

## Hotfix Releases

For urgent bug fixes:

1. Create hotfix branch from main
2. Fix the bug
3. Test thoroughly
4. Release as PATCH version
5. Merge back to main

## Release Schedule

- **Major releases**: Every 6-12 months
- **Minor releases**: Every 1-3 months
- **Patch releases**: As needed
- **Hotfixes**: As needed

## Automation

Consider automating:

- [ ] Running tests
- [ ] Building packages
- [ ] Uploading to PyPI
- [ ] Creating GitHub releases
- [ ] Updating documentation
- [ ] Sending notifications
