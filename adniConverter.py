#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script that prepares ADNI-data for use with C5.0 and/or WEKA.

Takes a folder of xml files (metadata) and subfolders with .nii files (MRIs),
as downloaded from LONI archives via webbrowser (Java applet).
May perform dimensionality reduction (depending on arguments), and outputs
spec- and data-files for use with systems supplied as arguments.
"""
__author__ = "Eivind Arvesen"
__copyright__ = "Copyright (c) 2014-2015, Eivind Arvesen. All rights reserved."
__credits__ = ["Eivind Arvesen"]  # Bugfixers, suggestions etc.
__license__ = "GPL3"  # LGPL/GPL3/BSD/Apache/MIT/CC/C/whatever
__version__ = "0.0.3 Alpha"
__maintainer__ = "Eivind Arvesen"
__email__ = "eivind.arvesen@gmail.com"
__status__ = "Prototype"  # Prototype/Development/Production
__date__ = "2015/03/31 03:43:40 CEST"
__todo__ = [
    "In some serious need of generalizations/modularizations...",
    "Create logging method, so that the script doesn't fail"
    "silently or just quit...",
    "Check/try/error that number of XML and NIFTI are alike?",
    "Check that no return values (from methods) are empty...",
]
__bug__ = "None"

# Copyright (c) 2014-2015 Eivind Arvesen. All rights Reserved.

from itertools import izip_longest
from scipy import ndimage as nd
from sys import exit
import argparse
import collections
import copy_reg
import cPickle as pickle
import errno
import glob
import fnmatch
import multiprocessing as mp
import nibabel as nib
import numpy as np
import os
import re
import sys
import types
import xml.etree.cElementTree as ET


def _reduce_method(m):
    """Make instance methods pickleable."""
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _reduce_method)


class AdniConverter(object):

    """
    Convert ADNI dataset (MRI, NIfTI) to various formats.

    DESCRIBE PARAMS, METHODS, ETC.
    """

    input_folder = None
    file_stem = None
    out_folder = {}
    outformat = None
    reduction = None
    reduction_dict = {"P": {"name": "PCA", "value": 20},
                      "H": {"name": "Histogram", "value": 32}}
    n_slices = None
    visits = None
    outfiles = []
    logging = True
    visits_dict = {
        0: "ADNI1 Screening",
        1: "ADNI1/GO Month 6",
        2: "ADNI1/GO Month 12",
        3: "ADNI1/GO Month 18",
        4: "ADNI1/GO Month 24",
        5: "ADNI1/GO Month 36",
        6: "No Visit Defined"
    }
    out_dict = {
        "C": {"filesuffix": ".data", "fileformat": "C5.0"},
        "D": {"filesuffix": ".log", "fileformat": "DEBUG"},
        "V": {"filesuffix": ".csv", "fileformat": "CSV"},
        "W": {"filesuffix": ".arff", "fileformat": "WEKA"}
    }
    new_resolution = (192, 192, 160)
    third_dim = None
    dx_groups = {}
    max_size = None
    merge = None
    merge_dict = [
        "Normal,MCI,LMCI,AD",
        "Normal,MCI,AD",
        "Normal,Other",
        "Other,MCI",
        "Other,AD"
    ]

    def __init__(self):
        """Initialize. Handle args, check that ouput-files does not exist."""
        np.seterr(divide='ignore', invalid='ignore')
        # fix PCA divide by zero "errors"...
        parser = argparse.ArgumentParser(description='A script for converting (a subset of) the ADNI1/GO MRI dataset to a format that is compatible with C5.0 or WEKA.\nOutputs to relative path "Converted/<attributes>"', prog='ADNI Converter', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='''\n
VISITS [-i] (ACCEPTED ARS, INCLUSIVE UPPER BOUND) | MERGED [-x] DX GROUPS
0 : ADNI1 Screening                               | 0: Normal / MCI / LMCI / AD
1 : ADNI1/GO Month 6                              | 1: Normal / MCI / AD
2 : ADNI1/GO Month 12                             | 2: Normal / Other
3 : ADNI1/GO Month 18                             | 3: Other / MCI
4 : ADNI1/GO Month 24                             | 4: Other / AD
5 : ADNI1/GO Month 36                             |
6 : No Visit Defined                              |
    ''')
        parser.add_argument(
            '-c', '--clean', help='remove any previous output',
            action='store_true', default=False)
        parser.add_argument(
            '-d', '--directory', required=True, help='directory to use',
            action='store')
        parser.add_argument(
            '-f', '--format', nargs='*', choices=['C', 'V', 'W'],
            default=['C', 'V'], help='Output format (C5.0, CSV, Weka)',
            action='store')
        parser.add_argument(
            '-g', '--getInfo',
            help='Show amount of images from visits and exit',
            action='store_true', default=False)
        parser.add_argument(
            '-i', '--visits', type=int,
            help='Latest visit to include (int <=6)', action='store')
        parser.add_argument(
            '-m', '--maxSize', type=float,
            help='Maximum output file size (in GiB)',
            action='store')
        parser.add_argument(
            '-n', '--slices', type=int, default=1, help='Use every Nth slice',
            action='store')
        parser.add_argument(
            '-r', '--reduction', choices=['P', 'H'], default=None,
            help='Method used for dimensionality reduction (PCA, Histogram)',
            action='store')
        parser.add_argument(
            '-s', '--scale', type=int, default=(192, 192, 160),
            help='Resolution to downscale to (x,y,z)', action='store', nargs=3)
        parser.add_argument(
            '-v', '--version', help='display version',
            action='version', version='%(prog)s ' + __version__)
        parser.add_argument(
            '-x', '--mergeGroups', type=int, nargs='*',
            help='Merge DX groups (int <=4)', action='store')
        self.args = parser.parse_args()

        self.input_folder = self.args.directory

        if self.args.getInfo:
            self.getInfo()
            sys.exit(0)

        if self.args.visits is not None:
            self.visits = self.args.visits
        else:
            if self.args.maxSize:
                self.visits = 6
            else:
                self.visits = 0

        if self.args.maxSize:
            self.max_size = self.args.maxSize

        self.new_resolution = tuple(self.args.scale)
        self.new_dimensions = self.args.scale

        if self.args.slices:
            self.n_slices = self.args.slices
            self.new_dimensions[0] = self.new_dimensions[0] / self.n_slices

        if self.args.reduction is not None:
            self.reduction = self.args.reduction
            self.new_dimensions[2] = self.reduction_dict[
                self.reduction]["value"]
            if self.reduction == "H":
                self.new_dimensions[1], self.new_dimensions[0] = 1, 1

        if self.args.mergeGroups is not None:
            self.merge = [x for x in self.args.mergeGroups]
        else:
            self.merge = [0]

        self.dx_groups = {x: {} for x in self.merge}

        self.file_stem = filter(lambda x: not re.match(
            r'^\s*$', x),
            [x.strip() for x in self.input_folder.split('/')])[-1]

        try:
            os.makedirs("Converted/")
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise IOError('Error encountered during file creation.')

        for mergeGroup in self.merge:
            # Bruk os.path.join() - forrige gang ble sep'en en del av
            # fil/mappe-navnet...
            self.out_folder.update({mergeGroup:
                                    "Converted/" + self.file_stem +
                                    "_visits-" + str(self.visits) +
                                    "_nSlices-" + str(self.n_slices) +
                                    "_reduction-" +
                                    str(self.reduction) + "_scale-" +
                                    str(self.new_resolution[0]) + "x" +
                                    str(self.new_resolution[1]) + "x" +
                                    str(self.new_resolution[2]) +
                                    "_mergeGroups-" + str(mergeGroup) + "/"})

            try:
                os.makedirs(self.out_folder[mergeGroup])
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise IOError('Error encountered during file creation.')

        self.outfiles = list(self.out_dict[x.upper()]['filesuffix'] for x in self.args.format)
        if ("C" in self.args.format):
            self.outfiles.append(".names")
        if ("D" not in self.args.format):
            self.outfiles.append(".log")

        if self.args.clean:
            for mergeGroup in self.merge:
                for outfile in self.outfiles:
                    for fl in glob.glob(self.out_folder[mergeGroup] +
                                        self.file_stem + outfile):
                        try:
                            os.remove(fl)
                        except OSError, e:
                            raise IOError('Could not remove previous files.')
            print "Removed any and all previous output files."

        for mergeGroup in self.merge:
            for outfile in self.outfiles:
                if os.path.isfile(self.out_folder[mergeGroup] +
                                  self.file_stem + outfile):
                    print "The file " + self.out_folder[mergeGroup] + \
                        self.file_stem + outfile + " already existed."
                    print "Please (re)move it before attempting to run this \
script again."
                    print "Exiting..."
                    exit(1)
                else:
                    with open(self.out_folder[mergeGroup] + self.file_stem +
                              outfile, 'w') as fout:
                        fout.write('')

            try:
                os.makedirs(self.out_folder[mergeGroup] + "Results/")
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise

        self.outformat = [x.upper() for x in self.args.format]

        print ""

        print 'Initializing...'
        dict_forms = []

        for format in self.outformat:
            dict_forms.append(self.out_dict[format]['fileformat'])

        ofs = ' + '.join(str(p) for p in dict_forms)
        print ofs + ' format(s) selected.\n'

        print "Using every", self.n_slices, "slice"
        try:
            print "Dimensional reduction via", \
                self.reduction_dict[self.reduction]["name"]
        except Exception, e:
            print "No dimensional reduction."
        print "Scaling all images to", self.new_resolution
        print "Merging DX groups to scheme(s):"
        for x in self.merge:
            print str(x) + ":", self.merge_dict[x]
        print ""

        # parseXml(TRY return{...}, else fail>log)
        print 'Scanning input folder for XML-files from visits up until', \
            self.visits_dict[self.visits], '...'

        self.allFiles = self.filterFiles(
            self.getXmlFilesInFolder(self.input_folder), self.visits)

        gib_const = 10**9  # 1024**3 # 2**30
        if self.max_size:
            maxSize = self.max_size * gib_const
            print "Will stop writing when largest output-file exceeds limit \
of approximately", self.__greek__(maxSize) + "B."
        else:
            maxSize = float("inf")

        cores = mp.cpu_count()
        # Detect number of logical (not physical; i.e. HyperThreading) cores
        print "\nDetected", cores, "(virtual/logical) cores."

        p = mp.Pool()
        manager = mp.Manager()
        q = manager.Queue()

        for mergeGroup in self.merge:
            if ("V" in self.outformat):
                sizeFile = self.out_folder[mergeGroup] + self.file_stem + \
                    self.out_dict['V']['filesuffix']
            if ("C" in self.outformat):
                sizeFile = self.out_folder[mergeGroup] + self.file_stem + \
                    self.out_dict['C']['filesuffix']
            if ("W" in self.outformat):
                sizeFile = self.out_folder[mergeGroup] + self.file_stem + \
                    self.out_dict['W']['filesuffix']

        writer = mp.Process(
            target=self.__outputProcess__, args=(q, sizeFile, maxSize, p))
        writer.start()

        for xmlFile in self.allFiles:
            p.apply_async(
                self.__workerProcess__, args=(xmlFile, q, sizeFile, maxSize))
        p.close()
        p.join()  # Wait for all child processes to close.
        writer.join()

        print 'CONVERSION DONE!\n'

    def __outputProcess__(self, queue, sizeFile, maxSize, pool):
        """Listen for messages on queue. Perform all writing of output."""
        print'\nWriting spec-files...'

        if ("V" in self.outformat):
            self.writeCsvHeader()
            print 'Wrote ".csv" header.'
        if ("C" in self.outformat):
            self.writeNames()
            print 'Wrote ".names" file.'
        if ("W" in self.outformat):
            self.writeArffHeader()
            print 'Wrote ".arff " file HEADER.'

        print '\nStarting conversion of', self.allFiles.__len__(), \
            'NIfTI images.\n'

        images_used = 0
        buffer = {}
        while 1:
            m = pickle.loads(queue.get())
            if m == 'STOP':
                # SHOULDN'T THIS BE POOL INSTEAD?
                p.terminate()
                break
            current_image = images_used + 1
            if m['file_index'] != current_image:
                # save queue object in buffer if not current index
                buffer[m['file_index']] = m
            else:
                if os.path.getsize(sizeFile) < maxSize:
                    self.writeLine(m['data'], m['file_object'])
                    print 'Converted and wrote image', m['file_index'], 'of',\
                        self.allFiles.__len__(), "- Largest size-constricted \
output-file", self.__greek__(os.path.getsize(sizeFile)) + "B /",\
                        self.__greek__(maxSize) + "B."
                    images_used += 1
                    if os.path.getsize(sizeFile) >= maxSize:
                        queue.put('STOP')
                        break
                    else:
                        current_image += 1
                        while current_image in buffer:
                            m = buffer[current_image]
                            self.writeLine(m['data'], m['file_object'])
                            print 'Converted and wrote image', m['file_index'], 'of', self.allFiles.__len__(), "- Largest size-constricted output-file", self.__greek__(os.path.getsize(sizeFile)) + "B /", self.__greek__(maxSize) + "B."
                            del buffer[current_image]
                            images_used += 1
                            current_image += 1
                            if os.path.getsize(sizeFile) >= maxSize:
                                queue.put('STOP')
                                break
                    if ("DEBUG" in self.outformat):
                        for mergeGroup in self.merge:
                            self.log(
                                mergeGroup, self.prettyFormat(
                                                    self.parseXml(xmlFile)))
                else:
                    queue.put('STOP')
                    break
            if current_image > self.allFiles.__len__():
                queue.put('STOP')
                break
        self.logRun(self.allFiles.__len__(), images_used)
        print "Processed and wrote", images_used, "files and lines in total."

    def __workerProcess__(self, xmlFile, queue, sizeFile, maxSize):
        """Perform all data processing. Write results to queue."""
        if os.path.getsize(sizeFile) < maxSize:
            filei = self.allFiles.index(xmlFile) + 1
            fileo = self.parseXml(xmlFile)
            result = {'data': self.processData(
                fileo), 'file_index': filei, 'file_object': fileo}
            if os.path.getsize(sizeFile) < maxSize:
                # pickle numpy-array as binary data to increase performance
                queue.put(pickle.dumps(result, protocol=-1))
            else:
                queue.put("STOP")
        else:
            queue.put("STOP")

    def __greek__(self, size):
        """Return a string representing the greek/metric suffix of a size."""
        # http://www.gossamer-threads.com/lists/python/python/18406
        abbrevs = [
            (1 << 50L, 'P'),
            (1 << 40L, 'T'),
            (1 << 30L, 'G'),
            (1 << 20L, 'M'),
            (1 << 10L, 'k'),
            (1, '')
        ]
        for factor, suffix in abbrevs:
            if size > factor:
                break
        return ("%.2f" % float((size / factor))) + ' ' + suffix

    def getInfo(self):
        """Output visits from ADNI."""
        visit_groups = {}
        dx_groups = {}

        print "Scanning input folder for XML-files..."
        xml_files = self.getXmlFilesInFolder(self.input_folder)

        print "Counting subjects in visit and diagnostic groups...\n"
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            visit = root.find(".//*visitIdentifier").text
            dx = root.find(".//*[@item='DX Group']").text

            if visit not in visit_groups:
                visit_groups.update({visit: 0})
            visit_groups.update({visit: (visit_groups[visit] + 1)})

            if dx not in dx_groups:
                dx_groups.update({dx: 0})
            dx_groups.update({dx: (dx_groups[dx] + 1)})

        i = 0

        visit_groups = collections.OrderedDict(
            sorted(visit_groups.items(), key=lambda x: (x[0][0:6], x[0][-2:])))
        dx_groups = collections.OrderedDict(
            reversed(sorted(dx_groups.items())))
        for e1, e2 in list(izip_longest(visit_groups.items(),
                                        dx_groups.items(), fillvalue=('', '')
                                        )):
            print i, ":", e1[0], "\t", e1[1], "\t\t\t  |  ", e2[0], "\t", e2[1]
            i += 1
        print ""

    def writeArffHeader(self):
        """Write header for Arff-file (WEKA)."""
        for mergeGroup in self.merge:
            with open(self.out_folder[mergeGroup] + self.file_stem + ".arff",
                      'w') as nfile:
                nfile.write('@RELATION ADNI\n\n')
                # nfile.write('@ATTRIBUTE ID string\n')
                # nfile.write('@ATTRIBUTE age numeric\n')
                # nfile.write('@ATTRIBUTE sex {M,F}\n')
                # nfile.write('@ATTRIBUTE "APOE A1" integer\n')
                # nfile.write('@ATTRIBUTE "APOE A2" integer\n')
                # nfile.write('@ATTRIBUTE "MMSE Total Score" numeric\n\n')

                # for number in range final resolution (192, 192, 160)
                # each slice reduced to X components and using every Nth slice:
                # THIS NEEDS TO TAKE VALUES FROM command line parameters etc.
                for number in range((self.new_dimensions[0]) *
                                    self.new_dimensions[1] *
                                    self.new_dimensions[2]):
                    nfile.write(
                        '@ATTRIBUTE "pixel ' + str(number + 1) + '" real\n')
                nfile.write(
                    '@ATTRIBUTE diagnosis {' + self.merge_dict[mergeGroup] +
                    '}\n')
                nfile.write('\n@DATA\n')

    def writeNames(self):
        """Write Names-file (C5.0)."""
        for mergeGroup in self.merge:
            with open(self.out_folder[mergeGroup] + self.file_stem + ".names",
                      'w') as nfile:
                nfile.write('diagnosis.              | target attribute\n\n')
                # nfile.write('ID:                     label.\n')
                # nfile.write('age:                    continuous.\n')
                # nfile.write('sex:                    M, F.\n')
                # nfile.write('APOE A1:                discrete 4.\n')
                # nfile.write('APOE A2:                discrete 4.\n')
                # nfile.write('MMSE Total Score:       continuous.\n\n')

                # for number in range final resolution (192, 192, 160),
                # each slice reduced to X components and using every Nth slice:
                # THIS NEEDS TO TAKE VALUES FROM command line parameters etc.
                nfile.write(
                    'diagnosis:              ' + self.merge_dict[mergeGroup] +
                    '.\n')
                for number in range((self.new_dimensions[0]) *
                                    self.new_dimensions[1] *
                                    self.new_dimensions[2]):
                    nfile.write(
                        'pixel ' + str(number + 1) + ':        continuous.\n')

    def writeCsvHeader(self):
        """Write header for CSV-file (?/Pylearn2)."""
        for mergeGroup in self.merge:
            with open(self.out_folder[mergeGroup] + self.file_stem + ".csv",
                      'w') as nfile:
                nfile.write('diagnosis,')
                for number in range((self.new_dimensions[0]) *
                                    self.new_dimensions[1] *
                                    self.new_dimensions[2]):
                    nfile.write('pixel_' + str(number + 1) + ',')
                nfile.seek(-1, os.SEEK_END)
                nfile.truncate()
                nfile.write('\n')

    def getXmlFilesInFolder(self, folder):
        """Get a list of all XML files in a folder (non-recursive search)."""
        xml_files = []
        for xml_file in os.listdir(folder):
            if xml_file.endswith(".xml"):
                xml_files.append(os.path.join(self.input_folder, xml_file))
        return xml_files

    def filterFiles(self, xmls, visits):
        """Filter out unwanted XML files, i.e. not within specified range."""
        print "Filtering through", len(xmls), "XMLs..."
        relevant_xmls = []

        if self.visits == 6:
            relevant_xmls = xmls

        else:
            for xf in xmls:
                tree = ET.parse(xf)
                root = tree.getroot()

                # if (root.find(".//*visitIdentifier").text not in visits):
                j = 0
                while j <= visits:
                    if(root.find(".//*visitIdentifier").text ==
                       self.visits_dict[j]):
                        relevant_xmls.append(xf)
                    j += 1

        print "Using", len(relevant_xmls), "XMLs."
        return relevant_xmls

    def parseXml(self, xml_file):
        """Get associated metadata and corresponding NIfTI file from XML."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        try:
            # 'ID': self.getId(root), 'Age': self.getAge(root),
            # 'Sex': self.getSex(root), 'APOE A1': self.getApoeA1(root),
            # 'APOE A2': self.getApoeA2(root),
            # 'MMSE Score': self.getMmseScore(root),
            return {'DX Group': self.getDxGroup(root),
                    'Nifti File': self.getNiftiFile(root)}
        except:
            e = sys.exc_info()[0]
            print "Error parsing:", e

    def getId(self, root):
        """Get subject ID from XML (root) element."""
        return root.find(".//*subjectIdentifier").text

    def getAge(self, root):
        """Get subject age from XML (root) element."""
        return root.find(".//*subjectAge").text

    def getSex(self, root):
        """Get subject sex from XML (root) element."""
        return root.find(".//*subjectSex").text

    def getApoeA1(self, root):
        """Get subject APOE A1 from XML (root) element."""
        return root.find(".//*[@item='APOE A1']").text

    def getApoeA2(self, root):
        """Get subject APOE A2 from XML (root) element."""
        return root.find(".//*[@item='APOE A2']").text

    def getMmseScore(self, root):
        """Get subject MMSE Total Score from XML (root) element."""
        return root.find(".//*[@attribute='MMSCORE']").text

    def getDxGroup(self, root):
        """Get subject diagnostic group from XML (root) element."""
        return root.find(".//*[@item='DX Group']").text

    def getNiftiFile(self, root):
        """Get corresponding NIfTI file from XML (root) element."""
        subjectIdentifier = root.find(".//*subjectIdentifier").text
        seriesIdentifier = root.find(".//*seriesIdentifier").text
        imageUID = root.find(".//*imageUID").text

        searchFor = 'ADNI_' + subjectIdentifier + '_*_' + 'S' + \
            seriesIdentifier + '_' + 'I' + imageUID + '.nii'

        matches = []
        for rootB, dirnames, filenames in os.walk(self.input_folder):
            for filename in fnmatch.filter(filenames, searchFor):
                matches.append(os.path.join(rootB, filename))
        if (matches.__len__() < 1):
            print 'There was no corresponding .nii match using the following \
pattern:'
            print searchFor
            print 'Exiting...'
            exit(1)
        elif (matches.__len__() > 1):
            print 'There was more than one corresponding .nii match using the \
following pattern:'
            print searchFor
            print 'Exiting...'
            exit(1)

        return matches[0]

    def mergeGroups(self, scheme, group):
        """Merge DX groups."""
        if scheme == 0 or scheme is None:
            pass

        elif scheme == 1:
            if group == 'LMCI':
                group = 'MCI'

        elif scheme == 2:
            if group != 'Normal':
                group = 'Other'

        elif scheme == 3:
            if group != 'MCI':
                if group == 'LMCI':
                    group = 'MCI'
                else:
                    group = 'Other'

        elif scheme == 4:
            if group != 'AD':
                group = 'Other'

        else:
            print "Failed to merge into another group."

        return group

    def prettyFormat(self, current_file):
        """Produce prettified ouput. For testing purposes."""
        # print "ID: ", current_file['ID']
        # print "Age: ", current_file['Age']
        # print "Sex: ", current_file['Sex']
        # print "APOE A1: ", current_file['APOE A1']
        # print "APOE A2", current_file['APOE A2']
        # print "MMSE Score: ", current_file['MMSE Score']
        print "DX Group: ", current_file['DX Group']
        print "Nifti File: ", current_file['Nifti File']

    def log(self, mergeGroup, message):
        """Write to log."""
        with open(self.out_folder[mergeGroup] + self.file_stem +
                  ".log", 'a') as logf:
            logf.write(str(message))

    def resize(self, img, new_resolution):
        """Resize (data from) 3D image matrix (numpy array)."""
        dsfactor = [w / float(f) for w, f in zip(new_resolution, img.shape)]
        new_img = nd.interpolation.zoom(img, zoom=dsfactor)
        return new_img

    def labelToInt(self, label, scheme):
        """Replace pre-merged label (string) with Int."""
        if scheme == 0 or scheme is None:
            if label == 'Normal':
                return '0'
            elif label == 'MCI':
                return '1'
            elif label == 'LMCI':
                return '2'
            elif label == 'AD':
                return '3'
            else:
                print "Failed to merge into another group."
                sys.exit(1)

        elif scheme == 1:
            if label == 'Normal':
                return '0'
            elif label == 'MCI':
                return '1'
            elif label == 'AD':
                return '2'
            else:
                print "Failed to merge into another group."
                sys.exit(1)

        elif scheme == 2:
            if label == 'Normal':
                return '0'
            elif label == 'Other':
                return '1'
            else:
                print "Failed to merge into another group."
                sys.exit(1)

        elif scheme == 3:
            if label == 'Other':
                return '0'
            elif label == 'MCI':
                return '1'
            else:
                print "Failed to merge into another group."
                sys.exit(1)

        elif scheme == 4:
            if label == 'Other':
                return '0'
            elif label == 'AD':
                return '1'
            else:
                print "Failed to merge into another group."
                sys.exit(1)

        else:
            print "Failed to merge into another group."
            sys.exit(1)

    def maybeReduceDimensionality(self, img_data):
        """Dimensional reduction of 3D image matrix (numpy array)."""
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        img_data = img_data[::self.n_slices]

        if self.reduction is None:
            """No Reduction"""
            return img_data

        elif self.reduction == "H":
            """Histogram"""
            from sklearn import preprocessing

            img_data = np.asarray(img_data, dtype=float).flat

            min_max_scaler = preprocessing.MinMaxScaler()
            scaled_data = min_max_scaler.fit_transform(img_data)

            hist = np.histogram(scaled_data,
                                bins=self.reduction_dict["H"]["value"],
                                range=None, normed=False, weights=None,
                                density=None)[0]

            return hist.reshape(1, hist.shape[0])

        elif self.reduction == "P":
            """Slice-wise (randomized) Principal Component Analysis"""
            from sklearn.preprocessing import normalize
            from sklearn.decomposition import RandomizedPCA

            proj_data = []
            for img_slice in img_data:
                norm_data = normalize(img_slice)

                shaped_data = np.reshape(norm_data, norm_data.size)
                # shaped_data.shape

                rpca = RandomizedPCA(
                    n_components=self.reduction_dict["P"]["value"],
                    random_state=0)
                proj_slice = rpca.fit_transform(norm_data)
                # plt.imshow(proj_data)

                # feat_data = rpca.inverse_transform(proj_data)
                # plt.imshow(feat_data)
                # plt.imshow(norm_data)

                proj_data.append(proj_slice)

            return proj_data

    def processData(self, current_file):
        """Process data."""
        return self.maybeReduceDimensionality(self.resize(nib.load(
            current_file['Nifti File']).get_data(), self.new_resolution))

    def writeLine(self, img_data, current_file):
        """Write image data as line in dataset file(s)."""
        for mergeGroup in self.merge:
            for format in self.outformat:
                output_file = self.out_folder[mergeGroup] + self.file_stem + \
                    self.out_dict[format]['filesuffix']
                output_format = self.out_dict[format]['fileformat']

                with open(output_file, "a") as myfile:
                    # myfile.write(current_file['ID'] + ',' + \
                    # current_file['Age'] + ',' + current_file['Sex'] + \
                    # ',' + current_file['APOE A1'] + ',' + \
                    # current_file['APOE A2'] + ',')

                    if (format != 'W'):
                        if (format == 'V'):
                            myfile.write(
                                str(self.labelToInt(self.mergeGroups(
                                    mergeGroup, current_file['DX Group']), mergeGroup)) +
                                ',')
                        else:
                            myfile.write(
                                self.mergeGroups(mergeGroup,
                                                 current_file['DX Group']) +
                                ',')

                    i = 0
                    for data_slice in img_data:
                        np.savetxt(myfile, (data_slice * (10**6)).astype(int),
                                   delimiter=",", newline=',', fmt="%d")  # s/f
                        i += 1

                    # hack to remove single (illegal[?]) comma on end of line
                    # (MAY NOT WORK ON ALL PLATFORMS [i.e. Windows])
                    myfile.seek(-1, os.SEEK_END)
                    myfile.truncate()
                    if (format == 'W'):
                        myfile.write(
                            ',' + self.mergeGroups(mergeGroup,
                                                   current_file['DX Group']))
                    myfile.write('\n')

            group = self.mergeGroups(mergeGroup, current_file['DX Group'])
            if group not in self.dx_groups[mergeGroup]:
                self.dx_groups[mergeGroup].update({group: 0})
            self.dx_groups[mergeGroup].update(
                {group: (self.dx_groups[mergeGroup][group] + 1)})

    def logRun(self, total_files, images_used):
        """Output relevant information from current conversion."""
        for mergeGroup in self.merge:
            self.log(mergeGroup, "CONVERSION INFORMATION:\n\n")
            dict_forms = []
            for format in self.outformat:
                dict_forms.append(self.out_dict[format]['fileformat'])
            ofs = ' + '.join(str(p) for p in dict_forms)
            self.log(
                mergeGroup, "Started out using " + str(total_files) +
                " .nii files.\n")
            if self.max_size is not None:
                self.log(mergeGroup,
                         "Stopped conversion when largest ouput file reached "
                         + str(self.max_size) + "GiB.\n")
            self.log(
                mergeGroup, "Wrote " + str(images_used) + " lines in total.\n")
            self.log(mergeGroup, "Resized all NIfTI MR Images to " +
                     str(self.new_resolution) +
                     " (lowest resolution in set).\n\n")
            self.log(mergeGroup, "Included visits:\n")
            for x in xrange(0, self.visits + 1):
                self.log(mergeGroup, self.visits_dict[x] + "\n")
            self.log(
                mergeGroup, "\nUsed every " + str(self.n_slices) +
                " slice(s).\n")
            if self.reduction is not None:
                self.log(mergeGroup, "Reduced dimensionality of each slice to "
                         + str(self.reduction_dict[self.reduction]["value"]) +
                         " components/bins via method " +
                         str(self.reduction_dict[self.reduction]["name"]) +
                         ".\n")
            self.log(mergeGroup, "Converted to " + ofs + " format(s).\n")
            self.log(mergeGroup, "Final resolution was (" +
                     str(self.new_dimensions[0]) +
                     ", " + str(self.new_dimensions[1]) +
                     ", " + str(self.new_dimensions[2]) +
                     ").\n")
            self.log(mergeGroup, "DX Groups after (eventual) merging: " +
                     self.merge_dict[mergeGroup] + ".\n")
            self.log(mergeGroup, "\nSubjects in diagnostic groups:\n")
            groups = collections.OrderedDict(
                sorted(self.dx_groups[mergeGroup].items(), key=lambda t: t[0]))
            for k, v in groups.iteritems():
                self.log(mergeGroup, k + "\t" + str(v) + "\n")


if __name__ == '__main__':
    """Run only if the script is called explicitly."""

    obj = AdniConverter()
