from dataclasses import dataclass
import numpy as np
from collections import Counter
from pprint import pprint

@dataclass(unsafe_hash=True)
class EDU:
    index: int = None
    text: str = None
    nuclearity: bool = None

class RST:
    def __init__(self, edus, scheme2edus, scheme2nuclearity) -> None:

        self.edus = edus  # in order
        self.scheme2edus = scheme2edus
        self.scheme2nuclearity = scheme2nuclearity

    def __len__(self):
        return len(self.scheme2edus)

    def __str__(self):
        text = ""
        for edu_ in self.edus:
            text += str(edu_.text.strip())
            text += " "
        return text.rstrip()

    @classmethod
    def from_data(cls, tokens, segments, relations):

        edus, index2scheme, scheme2edus, scheme2nuclearity = [], {}, {}, {}

        # first pass: populate edus
        for i, subtree in enumerate(relations):
            left_arg_edu_indices, right_arg_edu_indices, left_arg_nuclearity, right_arg_nuclearity, scheme = RST.parse_constituency_string(subtree)  # get rid of enclosing ( ))
            index2scheme[i] = scheme

            if len(left_arg_edu_indices) == 1:  # this is an EDU
                left_edu_index = left_arg_edu_indices[0]
                left_edu_token_start = 0 if left_edu_index == 1 else segments[left_edu_index - 2] + 1
                left_edu_token_end = segments[left_edu_index - 1] + 1
                left_edu_text = ''.join(tokens[left_edu_token_start:left_edu_token_end]).replace('▁', ' ')
                edus.append(EDU(left_edu_index, left_edu_text, left_arg_nuclearity))

            if len(right_arg_edu_indices) == 1:  # likewise
                right_edu_index = right_arg_edu_indices[0]
                right_edu_token_end = segments[right_edu_index - 1] + 1
                right_edu_token_start = 0 if right_edu_index == 1 else segments[right_edu_index - 2] + 1
                right_edu_text = ''.join(tokens[right_edu_token_start:right_edu_token_end]).replace('▁', ' ')
                edus.append(EDU(right_edu_index, right_edu_text, right_arg_nuclearity))

        edus = sorted(edus, key=lambda x: x.index)

        # second pass: mapping parents to children, parents to nuclearities
        for i, subtree in enumerate(relations):
            left_arg_edu_indices, right_arg_edu_indices, left_arg_nuclearity, right_arg_nuclearity, scheme = RST.parse_constituency_string(subtree)  # get rid of enclosing ( ))

            if len(left_arg_edu_indices) == 1 and len(right_arg_edu_indices) == 1:
                left_arg, right_arg = edus[left_arg_edu_indices[0] - 1], edus[right_arg_edu_indices[0] - 1]
                scheme2edus[str(i) + "-" + index2scheme[i]] = ([left_arg], [right_arg])

            elif len(left_arg_edu_indices) == 1 and len(right_arg_edu_indices) > 1:
                left_arg = edus[left_arg_edu_indices[0] - 1]
                right_arg = edus[right_arg_edu_indices[0] - 1 : right_arg_edu_indices[-1]]
                scheme2edus[str(i) + "-" + index2scheme[i]] = ([left_arg], right_arg)

            elif len(right_arg_edu_indices) == 1 and len(left_arg_edu_indices) > 1:
                right_arg = edus[right_arg_edu_indices[0] - 1]
                left_arg = edus[left_arg_edu_indices[0] - 1 : left_arg_edu_indices[-1]]
                scheme2edus[str(i) + "-" + index2scheme[i]] = (left_arg, [right_arg])

            else:
                left_arg = edus[left_arg_edu_indices[0] - 1 : left_arg_edu_indices[-1]]
                right_arg = edus[right_arg_edu_indices[0] - 1 : right_arg_edu_indices[-1]]
                scheme2edus[str(i) + "-" + index2scheme[i]] = (left_arg, right_arg)

            if left_arg_nuclearity and right_arg_nuclearity:
                scheme2nuclearity[str(i) + "-" + index2scheme[i]] = "NN"
            elif left_arg_nuclearity:
                scheme2nuclearity[str(i) + "-" + index2scheme[i]] = "NS"
            else:
                scheme2nuclearity[str(i) + "-" + index2scheme[i]] = "SN"

        return cls(edus, scheme2edus, scheme2nuclearity)
    
    @staticmethod
    def parse_constituency_string(subtree_string):
        subtree_string = subtree_string.strip()[1:-1]
        left_arg_parse, right_arg_parse = subtree_string.split(",")
        left_arg_nuclearity, right_arg_nuclearity = "Nucleus" in left_arg_parse, "Nucleus" in right_arg_parse
        left_arg_start, left_arg_end = int(left_arg_parse.split(":")[0]), int(left_arg_parse.split(":")[-1])
        right_arg_start, right_arg_end = int(right_arg_parse.split(":")[0]), int(right_arg_parse.split(":")[-1])
        left_arg_edu_indices, right_arg_edu_indices = list(range(left_arg_start, left_arg_end + 1)), list(range(right_arg_start, right_arg_end + 1))
        left_arg_scheme, right_arg_scheme = left_arg_parse.split("=")[-1].split(":")[0], right_arg_parse.split("=")[-1].split(":")[0]
        scheme = left_arg_scheme if left_arg_scheme != "span" else right_arg_scheme
        return left_arg_edu_indices, right_arg_edu_indices, left_arg_nuclearity, right_arg_nuclearity, scheme

    @staticmethod
    def sample_marcu99(rst):
        
        p1, p2 = RST.get_most_salient_nucleus(rst)

        if p1[0].index < p2[0].index:
            return p1[0].text, p2[0].text
        else:
            return p2[0].text, p1[0].text

    
    @staticmethod
    def get_most_salient_nucleus(rst, n=2):

        root = sorted(rst.scheme2edus.keys())[0]
        scheme = root

        if rst.scheme2nuclearity[scheme] == "NN": # search left and right
            candidates = [e for e in rst.scheme2edus[scheme][0] if e.nuclearity] + [e for e in rst.scheme2edus[scheme][1] if e.nuclearity]
        elif rst.scheme2nuclearity[scheme] == "NS": # search left
            candidates = [e for e in rst.scheme2edus[scheme][0] if e.nuclearity]
        else: # search right
            candidates = [e for e in rst.scheme2edus[scheme][1] if e.nuclearity]

        raw = RST.recur_most_salient(rst, 0, candidates)
        candidate2freq = Counter(raw)
        return candidate2freq.most_common(n)
        

    @staticmethod
    def recur_most_salient(rst, seen, candidates):
        rstnode = sorted(rst.scheme2edus.keys())[seen]
        scheme = rstnode

        while True:
            if len(candidates) == 1 or seen == len(rst.scheme2edus.keys()) - 1:
                return candidates
            
            if rst.scheme2nuclearity[scheme] == "NN": # search left and right
                potential = [e for e in rst.scheme2edus[scheme][0] if e.nuclearity] + [e for e in rst.scheme2edus[scheme][1] if e.nuclearity]
                potential = candidates + [p for p in potential if p in candidates]
                return RST.recur_most_salient(rst, seen + 1, potential)
            elif rst.scheme2nuclearity[scheme] == "NS": # search left
                potential = [e for e in rst.scheme2edus[scheme][0] if e.nuclearity]
                potential = candidates + [p for p in potential if p in candidates]
                return RST.recur_most_salient(rst, seen + 1, potential)
            else:
                potential = [e for e in rst.scheme2edus[scheme][1] if e.nuclearity]
                potential = candidates + [p for p in potential if p in candidates]
                return RST.recur_most_salient(rst, seen + 1, potential)

    @staticmethod
    def apply_simclr_transforms(rst, p=0.5):
        # first transform is always a random cutout, biased towards the top of the tree for longer anchors
        # second transform can be random (edu/nucleus/satellite) dropout, edu (rotation/shuffle)
        # transform fn indices : [rand_edu_dropout, nucleus_dropout, satellite_dropout, rand_edu_rotation, shuffle_edus]
        ###################### : [       0                 1                  2                  3                 4]
        
        all_left_args, all_right_args, all_schemes = RST.all_rst_cutouts(rst_example)
        all_cutouts_valid = [(len(left_) + len(right_), left_, right_, scheme_) for left_, right_, scheme_ in zip(all_left_args, all_right_args, all_schemes) if len(left_) + len(right_) > 3]
        _, left_arg_edus, right_arg_edus, _ = sorted(all_cutouts_valid, key=lambda x:x[0])

        n_subtrees = len(all_cutouts_valid)
        random_crop_index_1, random_crop_index_2 = np.random.randint(0, n_subtrees // 2, size=2)
        edu_span_1 = left_arg_edus[random_crop_index_1].extend(right_arg_edus[random_crop_index_1])
        edu_span_2 = left_arg_edus[random_crop_index_2].extend(right_arg_edus[random_crop_index_2])

        tx_1, tx_2 = [], []
        random_augm_index_1 = np.random.randint(0, 5)
        random_augm_index_2 = np.random.randint(0, 5)

        # branch 1
        if random_augm_index_1 == 0:
            # random dropout
            for e in edu_span_1:
                if p < np.random.rand(1):
                    tx_1.append(e)

        elif random_augm_index_1 == 1:
            # nucleus dropout
            for e in edu_span_1:
                if not e.nuclearity: # keep all satellite
                    tx_1.append(e)
                elif float(3*p/2) < np.random.rand(1):
                    tx_1.append(e)

        elif random_augm_index_1 == 2:
            # satellite dropout
            for e in edu_span_1:
                if e.nuclearity: # keep all nuclei
                    tx_1.append(e)
                elif float(3*p/2) < np.random.rand(1):
                    tx_1.append(e)
        
        elif random_augm_index_1 == 3:
            # rotation
            random_k = np.random.randint(1, len(edu_span_1) // 2)
            tx_1 = edu_span_1[random_k:] + edu_span_1[:random_k]
        
        else: # 4
            # shuffle
            tx_1 = np.random.shuffle(edu_span_1)

        # branch 2
        if random_augm_index_2 == 0:
            # random dropout
            for e in edu_span_2:
                if p < np.random.rand(1):
                    tx_2.append(e)

        elif random_augm_index_2 == 1:
            # nucleus dropout
            for e in edu_span_2:
                if not e.nuclearity: # keep all satellite
                    tx_2.append(e)
                elif float(3*p/2) < np.random.rand(1):
                    tx_2.append(e)

        elif random_augm_index_2 == 2:
            # satellite dropout
            for e in edu_span_2:
                if e.nuclearity: # keep all nuclei
                    tx_2.append(e)
                elif float(3*p/2) < np.random.rand(1):
                    tx_2.append(e)
        
        elif random_augm_index_2 == 3:
            # rotation
            random_k = np.random.randint(1, len(edu_span_2) // 2)
            tx_2 = edu_span_2[random_k:] + edu_span_2[:random_k]
        
        else: # 4
            # shuffle
            tx_2 = np.random.shuffle(edu_span_2)

        return " ".join([e.text for e in edu_span_1]), " ".join([e.text for e in edu_span_2])

    @staticmethod
    def sample_3deep(rst):
        left_edus, right_edus = RST.get_left_subtree_edus(rst), RST.get_right_subtree_edus(rst)
        left_span, right_span = " ".join(left_edus), " ".join(right_edus)
        longest = max(len(left_span), len(right_span))
        if not abs(len(left_span) - len(right_span)) / longest < 0.50:
            raise Exception

        # root, root-left, root-right
        a1 = left_span + " " + right_span
        p1, p2 = left_span, right_span

        # root-left, root-left-left, root-left-right
        a2 = left_span
        p3, p4 = " ".join([e for e in left_edus[:len(left_edus)//2]]), " ".join([e for e in left_edus[len(left_edus)//2:]])
        
        # root-right, root-right-left, root-right-right
        a3 = right_span
        p5, p6 = " ".join([e for e in right_edus[:len(right_edus)//2]]), " ".join([e for e in right_edus[len(right_edus)//2:]])

        a4 = " ".join([e for e in left_edus[:len(left_edus)//2]])
        p7, p8 = " ".join([e for e in left_edus[:len(left_edus)//4]]), " ".join([e for e in left_edus[len(left_edus)//4:len(left_edus)//2]])

        a5 = " ".join([e for e in left_edus[len(left_edus)//2:]])
        p9, p10 = " ".join([e for e in left_edus[len(left_edus)//2:3*len(left_edus)//4]]), " ".join([e for e in left_edus[3*len(left_edus)//4:]])

        a6 = " ".join([e for e in right_edus[:len(right_edus)//2]])
        p11, p12 = " ".join([e for e in right_edus[:len(right_edus)//4]]), " ".join([e for e in right_edus[len(right_edus)//4:len(right_edus)//2]])

        a7 = " ".join([e for e in right_edus[len(right_edus)//2:]])
        p13, p14 = " ".join([e for e in right_edus[len(right_edus)//2:3*len(right_edus)//4]]), " ".join([e for e in right_edus[3*len(right_edus)//4:]])

        return [
            {'anchor' : a2, 'left' : p3, 'right': p4},
            {'anchor' : a3, 'left' : p5, 'right': p6},
            {'anchor' : a4, 'left' : p7, 'right': p8},
            {'anchor' : a5, 'left' : p9, 'right': p10},
            {'anchor' : a6, 'left' : p11, 'right': p12},
            {'anchor' : a7, 'left' : p13, 'right': p14}
        ]

    @staticmethod
    def sample_2deep(rst):
        left_edus, right_edus = RST.get_left_subtree_edus(rst), RST.get_right_subtree_edus(rst)
        left_span, right_span = " ".join(left_edus), " ".join(right_edus)
        longest = max(len(left_span), len(right_span))
        if not abs(len(left_span) - len(right_span)) / longest < 0.50:
            raise Exception

        a1 = left_span + " " + right_span
        p1, p2 = left_span, right_span

        a2 = left_span
        p3, p4 = " ".join([e for e in left_edus[:len(left_edus)//2]]), " ".join([e for e in left_edus[len(left_edus)//2:]])

        a3 = right_span
        p5, p6 = " ".join([e for e in right_edus[:len(right_edus)//2]]), " ".join([e for e in right_edus[len(right_edus)//2:]])

        return [{'anchor' : a1, 'left' : p1, 'right' : p2}, {'anchor' : a2, 'left' : p3, 'right' : p4}, {'anchor' : a3, 'left' : p5, 'right' : p6}]

    @staticmethod
    def get_right_subtree(rst):
        for scheme in rst.scheme2edus:
            if '0' in scheme:
                right_span = " ".join([edu.text for edu in rst.scheme2edus[scheme][1]])
                return right_span
        return None

    @staticmethod
    def get_left_subtree(rst):
        for scheme in rst.scheme2edus:
            if '0' in scheme:
                left_span = " ".join([edu.text for edu in rst.scheme2edus[scheme][0]])
                return left_span
        return None

    @staticmethod
    def get_left_subtree_edus(rst):
        for scheme in rst.scheme2edus:
            if '0' in scheme:
                left_edus = [edu.text for edu in rst.scheme2edus[scheme][0]]
                return left_edus
        return None

    @staticmethod
    def get_right_subtree_edus(rst):
        for scheme in rst.scheme2edus:
            if '0' in scheme:
                right_edus = [edu.text for edu in rst.scheme2edus[scheme][1]]
                return right_edus
        return None

    @staticmethod
    def rotate_rst_edus(rst):
        n_edus = len(rst.edus)
        random_k = np.random.randint(1, n_edus // 2)
        rotated_edus = rst.edus[random_k:] + rst.edus[:random_k]
        return " ".join([edu.text for edu in rotated_edus])

    @staticmethod
    def all_rst_cutouts(rst):
        left_args, right_args, schemes = [], [], []
        for scheme in rst.scheme2edus:
            left_args.append(rst.scheme2edus[scheme][0])
            right_args.append(rst.scheme2edus[scheme][1])
            schemes.append(scheme)
        return left_args, right_args, schemes

    @staticmethod
    def two_random_rst_cutouts(rst):
        examples = []
        n_subtrees = len(rst.scheme2edus)
        random_indices = np.random.randint(1, n_subtrees, size=2)
        random_index_1, random_index_2 = str(random_indices[0]), str(random_indices[1])
        # random_index_1 = str(np.random.randint(1, n_subtrees))
        # random_index_2 = str(np.random.randint(n_subtrees // 2, n_subtrees - 1))
        for scheme in rst.scheme2edus:
            if random_index_1 in scheme or random_index_2 in scheme:
                left_span = " ".join([edu.text for edu in rst.scheme2edus[scheme][0]])
                right_span = " ".join([edu.text for edu in rst.scheme2edus[scheme][1]])
                parent = left_span + right_span
                examples.append({"parent" : parent, "left" : left_span, "right" : right_span})
                # if return_str:
                #     return str(left_span + " " + right_span)
                # return left_span, right_span, scheme.split("-")[-1]
        return examples

    @staticmethod
    def random_rst_cutout(rst, return_str=False):
        n_subtrees = len(rst.scheme2edus)
        random_index = str(np.random.randint(1, n_subtrees // 2))
        for scheme in rst.scheme2edus:
            if random_index in scheme:
                left_span = " ".join([edu.text for edu in rst.scheme2edus[scheme][0]])
                right_span = " ".join([edu.text for edu in rst.scheme2edus[scheme][1]])
                if return_str:
                    return str(left_span + " " + right_span)
                return left_span, right_span, scheme.split("-")[-1]
        return None

    @staticmethod
    def random_satellite_dropout(rst, p=0.2, return_str=False):
        if return_str:
            text = ""
            for edu_index, edu in enumerate(rst.edus):
                if edu.nuclearity:
                    text += str(edu.text.strip())
                    text += " "
                else:
                    if p < np.random.rand(1):
                        text += str(edu.text.strip())
                        text += " "

            return text.rstrip()
        elif p == 1:
            return [e.text for e in rst.edus if e.nuclearity]
        else:
            result = []
            for e in rst.edus:
                if e.nuclearity or p < np.random.rand(1):
                    result.append(str(e.text.strip()))
            return result

    @staticmethod
    def random_nucleus_dropout(rst, p=0.2, return_str=False):
        if return_str:
            text = ""
            for edu_index, edu in enumerate(rst.edus):
                if not edu.nuclearity:
                    text += str(edu.text.strip())
                    text += " "
                else:
                    if p < np.random.rand(1):
                        text += str(edu.text.strip())
                        text += " "

            return text.rstrip()
        elif p == 1:
            return [e.text for e in rst.edus if not e.nuclearity]
        else:
            result = []
            for e in rst.edus:
                if not e.nuclearity or p < np.random.rand(1):
                    result.append(str(e.text.strip()))
            return result

    @staticmethod
    def random_edu_dropout(rst, p=0.3, return_str=False):
        if return_str:
            text = ""
            for edu_index, edu in enumerate(rst.edus):
                if p < np.random.rand(1):
                    text += str(edu.text.strip())
                    text += " "
            return text.strip()
        else:
            result = []
            for e in rst.edus:
                if p < np.random.rand(1):
                    result.append(str(e.text.strip()))
            return result