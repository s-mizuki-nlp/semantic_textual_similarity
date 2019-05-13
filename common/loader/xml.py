#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Tuple, List, Optional, Iterable, Union
import sys, io, os
from lxml import etree
from lxml.etree import _Element, _ElementTree

class WSDDatasetLoader(object):

    __AVAILABLE_RETURN_TYPE = ("tokens","attrs")
    __ATTRIBUTES = "surface,pos,lemma,sensekey".split(",")
    __ATTRIBUTE_INDEX = {attr:idx for idx, attr in enumerate(__ATTRIBUTES)}

    def __init__(self, train_file_path, gold_file_path, return_type="tokens", xpath_sentence="//sentence", verbose=False):
        assert return_type in self.__AVAILABLE_RETURN_TYPE, "`return_type` must be one of these:" + ",".join(self.__AVAILABLE_RETURN_TYPE)
        self._path_train = train_file_path
        self._path_gold = gold_file_path
        self._return_type = return_type
        self._xpath_sentence = xpath_sentence
        self._verbose = verbose

        self._instance_to_sensekey = self._load_gold_set()
        self._xml_trainset = self._load_train_set()

    def __iter__(self):
        sentences = self._xml_trainset.xpath(self._xpath_sentence)

        if self._return_type == "tokens":
            for sentence in sentences:
                yield self._parse_sentence_to_token_sequence(sentence)
        elif self._return_type == "attrs":
            for sentence in sentences:
                yield self._parse_sentence_to_seqs_of_token_attrs(sentence)
        else:
            raise NotImplementedError(f"unexpected return type: {self._return_type}")

    def __len__(self):
        return len(self._xml_trainset.xpath(self._xpath_sentence))

    @property
    def n_example(self):
        return self.__len__()

    @property
    def n_annotation(self):
        sensekey_idx = self.get_attribute_index(attr_name="sensekey")
        sentences = self._xml_trainset.xpath(self._xpath_sentence)
        n_anno = 0
        for sentence in sentences:
            tup_lst_attr = self._parse_sentence_to_seqs_of_token_attrs(sentence)
            lst_seneskey = tup_lst_attr[sensekey_idx]
            n_anno += len(list(filter(bool, lst_seneskey)))
        return n_anno

    @property
    def ATTRIBUTES(self):
        return self.__ATTRIBUTES

    @property
    def ATTRIBUTE_INDEX(self):
        return self.__ATTRIBUTE_INDEX

    def get_attribute_index(self, attr_name: str):
        return self.__ATTRIBUTE_INDEX.get(attr_name, None)

    def token_attributes_to_dict(self, tup_token_attrs):
        return dict((attr, value) for attr, value in zip(self.__ATTRIBUTES, tup_token_attrs) )

    def _load_gold_set(self) -> Dict[str, str]:
        dict_instance_to_sense_key = {}
        n_multiple = 0
        with io.open(self._path_gold, mode="r") as ifs:
            for example in ifs:
                lst_entity = example.strip().split(" ")
                if len(lst_entity) > 2:
                    n_multiple += 1
                instance_id, sense_key = lst_entity[0], lst_entity[1]
                dict_instance_to_sense_key[instance_id] = sense_key

        if self._verbose:
            print(f"instances: {len(dict_instance_to_sense_key)}")
            print(f"instances with multi sense keys: {n_multiple}")

        return dict_instance_to_sense_key

    def _load_train_set(self) -> _ElementTree:
        return etree.parse(self._path_train)

    def _parse_token(self, xml_node_token: _Element) -> Tuple[Optional[str]]:
        token_type = xml_node_token.tag
        surface = xml_node_token.text
        pos = xml_node_token.attrib["pos"]
        lemma = xml_node_token.attrib["lemma"]
        if token_type == "instance":
            instance_id = xml_node_token.attrib["id"]
            sensekey = self._instance_to_sensekey.get(instance_id, None)
            if sensekey == None:
                raise KeyError(f"undefined instance id was found: {instance_id}")
        else:
            sensekey = None

        return (surface, pos, lemma, sensekey)

    def _parse_sentence_to_token_sequence(self, xml_node_sentence: _Element) -> List[Tuple[str]]:
        ret = list(map(self._parse_token, xml_node_sentence.getchildren()))
        return ret

    def _parse_sentence_to_seqs_of_token_attrs(self, xml_node_sentence: _Element) -> Tuple[Tuple[str]]:
        ret = tuple(zip(*map(self._parse_token, xml_node_sentence.getchildren())))
        return ret


class MinibatchWSDDatasetLoader(WSDDatasetLoader):

    def __init__(self, train_file_path: str, gold_file_path: str, n_minibatch: int,
                 return_type="tokens", xpath_sentence="//sentence", verbose=False):
        super(MinibatchWSDDatasetLoader, self).__init__(train_file_path, gold_file_path, return_type, xpath_sentence, verbose)
        self._n_mb = n_minibatch

    def __iter__(self):
        iter_example = super(MinibatchWSDDatasetLoader, self).__iter__()
        lst_ret = []
        for example in iter_example:
            lst_ret.append(example)
            if len(lst_ret) >= self._n_mb:
                yield lst_ret
                lst_ret = []
        if len(lst_ret) > 0:
            yield lst_ret