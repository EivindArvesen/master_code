#!/usr/bin/env bash

echo "Number of NIfTI files in (sub)folder(s): " $(find ADNI -name "*.nii" | wc -l)
