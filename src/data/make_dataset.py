# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
from collections import defaultdict


def group_docs(directory):
    """
    Groups documents by their name and annotators in the given directory

    :param directory:
    :return:
    """

    groups = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            basename, extension = os.path.splitext(filename)
            domain, id1, id2, annotator = basename.split('_')
            groups[domain + '_' + id1 + '_' + id2].append(filename)
    return groups


def process_documents(directory, base_info, file_list):
    """

    :param directory:
    :param base_info:
    :param file_list:
    :return:
    """

    mydict = {}

    base_infos = base_info.split('_')
    domain = base_infos[0]
    doc_id = base_infos[1] + '_' + base_infos[2]

    for file in sorted(file_list):
        count = 1
        with open(directory + '/' + file, 'r') as f:
            for line in f:
                key = str([domain, doc_id, count])
                value = []

                if "### abstract ###" in line:
                    section = 'abstract'
                    continue
                elif '### introduction ###' in line:
                    section = 'introduction'
                    continue
                elif mydict.has_key(key):
                    the_value = mydict.get(key)
                    # append the new number to the existing array at this slot
                    label, sentence = split_line(line)
                    the_value.append(label)
                    if len(the_value[1]) < len(sentence):
                        the_value[1] = sentence
                    count = count + 1
                else:
                    label, sentence = split_line(line)
                    value.append(section)
                    value.append(sentence)
                    value.append(label)
                    mydict[key] = value
                    count = count + 1

    return mydict


def split_line(line):

    line = line.replace('\t', ' ')
    line = line.replace('--', ' ')
    splitted_line = line.split(' ', 1)

    label = splitted_line[0].upper().strip()
    sentence = splitted_line[1].strip()

    return label, sentence


def save_to_file(articles, output_filepath):
    """

    :param articles:
    :param output_filepath:
    :return:
    """

    output = open(output_filepath, 'a')
    for key, values in sorted(articles.items()):

        key_string = key.replace('u\'', '').replace('[', '').replace(']', '\t').replace('\'', '').replace(',', '\t')
        value_string = '\t'.join(values)
        output.write(key_string + value_string + '\n')


@click.command()
@click.argument('input_folderpath', envvar='LABELED_ARTICLES_PATH', type=click.Path(exists=True))
@click.argument('output_filepath', envvar='FINAL_DATASET_PATH', type=click.Path(exists=False))
def main(input_folderpath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    groups = group_docs(input_folderpath)
    for key, value in groups.iteritems():
        article = process_documents(input_folderpath, key, value)
        save_to_file(article, output_filepath)

        logger.info('final data set generated')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
