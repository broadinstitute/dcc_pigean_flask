#python ~/lap/projects/ldsc/bin/factor_graph.py --gene-factors-in ../results/out/projects/eval/traits/T2D/results/result_mouse_msigdb_nohp_huge_gwas_detect/factors/factor_flat_gene_combined1/factor_flat_gene_combined1.gene_factors.out --gene-set-factors-in ../results/out/projects/eval/traits/T2D/results/result_mouse_msigdb_nohp_huge_gwas_detect/factors/factor_flat_gene_combined1/factor_flat_gene_combined1.gene_set_factors.out --factors-in ../results/out/projects/eval/traits/T2D/results/result_mouse_msigdb_nohp_huge_gwas_detect/factors/factor_flat_gene_combined1/factor_flat_gene_combined1.factors.out --X-in ../results/out/projects/eval/gene_set_lists/gene_set_list_mouse_2024.txt --X-in ../results/out/projects/eval/gene_set_lists/gene_set_list_msigdb_nohp.txt  --factor-col-prefix Factor --pdf-out ~/private_html/test.pdf  --pdf-bipartite-out ~/private_html/test2.pdf

#THIS CODE WAS GENERATED MOSTLY BY GPT-4o

import optparse
import sys
import os
import copy
import numpy as np
import gzip

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def bail(message):
    sys.stderr.write("%s\n" % (message))
    sys.exit(1)

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

usage = "python factor_graph.py"

parser = optparse.OptionParser(usage)

#Input columns from the factors file (used to define the mechanism/square nodes)
parser.add_option("","--factors-in",default=None)
parser.add_option("","--factors-id-col",default=None)
parser.add_option("","--factors-label-col",default=None)
parser.add_option("","--factors-beta-col",default=None)

parser.add_option("","--factors-anchor-in",default=None)
parser.add_option("","--factors-anchor-col",default="anchor")
parser.add_option("","--factors-relevance-col",default="relevance")

#Input columns from the gene factors file (used to define the gene/circle nodes)
parser.add_option("","--gene-factors-in",action='append',default=None)
parser.add_option("","--gene-factors-id-col",default=None)
parser.add_option("","--gene-factors-combined-col",default=None)
parser.add_option("","--gene-factors-direct-col",default=None)
parser.add_option("","--gene-factors-indirect-col",default=None)

parser.add_option("","--gene-anchor-factors-in",action='append',default=None)
parser.add_option("","--gene-anchor-factors-anchor-col",default="anchor")
parser.add_option("","--gene-anchor-factors-relevance-col",default="relevance")

#Input columns from the gene factors file (used to define the pheno/triangle nodes)
parser.add_option("","--pheno-factors-in",action='append',default=None)
parser.add_option("","--pheno-factors-id-col",default=None)
parser.add_option("","--pheno-factors-combined-col",default=None)
parser.add_option("","--pheno-factors-direct-col",default=None)
parser.add_option("","--pheno-factors-indirect-col",default=None)

parser.add_option("","--pheno-anchor-factors-in",action='append',default=None)
parser.add_option("","--pheno-anchor-factors-anchor-col",default="anchor")
parser.add_option("","--pheno-anchor-factors-relevance-col",default="relevance")

#Input columns from the gene pheno file (used to define edges between the gene and pheno nodes)
parser.add_option("","--gene-pheno-in",default=None)
parser.add_option("","--gene-pheno-gene-col",default=None)
parser.add_option("","--gene-pheno-pheno-col",default=None)
parser.add_option("","--gene-pheno-weight-col",default=None)

#Controls filters on which genes and gene/mechanism links are loaded
parser.add_option("","--filter-to-factor-genes",default=False,action='store_true')
parser.add_option("","--keep-zero-factors",default=False,action='store_true')
parser.add_option("","--min-num-nodes",default=0,type='float') #adjust combined threshold 
parser.add_option("","--gene-min-combined",default=None,type='float')
parser.add_option("","--gene-min-indirect",default=None,type='float')
parser.add_option("","--gene-min-loading",default=0.01,type='float')
parser.add_option("","--pheno-min-combined",default=None,type='float')
parser.add_option("","--pheno-min-indirect",default=None,type='float')
parser.add_option("","--pheno-min-loading",default=0.005,type='float')

parser.add_option("","--max-num-gene-nodes-per-factor",default=5,type='int')
parser.add_option("","--max-num-pheno-nodes-per-factor",default=5,type='int')

#Control of abstract node/edge sizes

parser.add_option("","--node-gene-loading-to-weight-pow",type=float,default=3) #in determining gene weights from their loadings, multiply the loading by this value
parser.add_option("","--node-pheno-loading-to-weight-pow",type=float,default=3) #in determining gene weights from their loadings, multiply the loading by this value
parser.add_option("","--node-size-scale",type=float,default=2) #multiply log_bf (genes) or beta (gene sets) by this number to get node size in graph.
parser.add_option("","--node-label-font-size",type=float,default=20) #multiply log_bf (genes) or beta (gene sets) by this number to get node size in graph.
parser.add_option("","--node-border-width",type=float,default=4) #outer border width; this is always the full color of the node and can be compared to the 
parser.add_option("","--node-label-size-threshold",type=float,default=10) #minimum node size for plotting labels
parser.add_option("","--mech-node-full-size",type=float,default=None) #cap out mechanism nodes at max size when hit this value
parser.add_option("","--mech-node-further-scale",type=float,default=1) #scale mechanism nodes by this additional factor
parser.add_option("","--mech-node-min-size",type=float,default=None) #cap out mechanism nodes at max size when hit this value
parser.add_option("","--pheno-node-full-size",type=float,default=None) #cap out mechanism nodes at max size when hit this value
parser.add_option("","--pheno-node-min-size",type=float,default=None) #scale mechanism nodes by this additional factor
parser.add_option("","--pheno-node-full-color",type=float,default=4) #cap out mechanism nodes at max size when hit this value
parser.add_option("","--pheno-edge-full-width",type=float,default=None) #cap out mechanism nodes at max size when hit this value
parser.add_option("","--pheno-node-further-scale",type=float,default=1) #scale mechanism nodes by this additional factor
parser.add_option("","--gene-node-full-size",type=float,default=None) #scale mechanism nodes by this additional factor
parser.add_option("","--gene-node-min-size",type=float,default=None) #scale mechanism nodes by this additional factor
parser.add_option("","--gene-node-full-color",type=float,default=None) #cap out mechanism nodes at max size when hit this value
parser.add_option("","--gene-edge-full-width",type=float,default=None) #scale mechanism nodes by this additional factor
parser.add_option("","--gene-node-further-scale",type=float,default=1) #scale mechanism nodes by this additional factor
parser.add_option("","--gene-pheno-edge-full-width",type=float,default=None) #scale mechanism nodes by this additional factor
parser.add_option("","--edge-max-width",type=float,default=5) #maximum edge width; multiply by node weight to get the final width
parser.add_option("","--edge-opacity",type=float,default=0.5) #maximum edge width; multiply by node weight to get the final width
parser.add_option("","--pheno-node-opacity",type=float,default=0.6) #maximum edge width; multiply by node weight to get the final width
parser.add_option("","--gene-node-opacity",type=float,default=1) #maximum edge width; multiply by node weight to get the final width
parser.add_option("","--coordinate-scale",type=float,default=5) #coordinate scale factor for PDF writing

parser.add_option("","--no-physics",action="store_false",default=True, dest="use_physics") #enable physics
parser.add_option("","--add-links",action="store_true") #add links

parser.add_option("","--colors-red-blue",action="store_true") #add links

parser.add_option("","--include-genes",type="string",action="callback",callback=get_comma_separated_args,default=None)
parser.add_option("","--include-phenos",type="string",action="callback",callback=get_comma_separated_args,default=None)

#Control of pdf and html plotting parameters
parser.add_option("","--pdf-width",type=float,default=50)
parser.add_option("","--pdf-height",type=float,default=50)
parser.add_option("","--html-height",type=int,default=1000)
parser.add_option("","--html-pos-scale",type=float,default=200)
parser.add_option("","--html-edge-width-scale",type=float,default=0.6)
parser.add_option("","--html-node-size-scale",type=float,default=0.002)
parser.add_option("","--html-node-border-scale",type=float,default=0.75)

#Control of output file locations
parser.add_option("","--log-file",default=None)
parser.add_option("","--warnings-file",default=None)
parser.add_option("","--debug-level",type='int',default=None)
parser.add_option("","--html-out",default=None)
parser.add_option("","--json-out",default=None)
parser.add_option("","--pdf-out",default=None)

(options, args) = parser.parse_args()

#set up warnings
warnings_fh = None
if options.warnings_file is not None:
    warnings_fh = open(options.warnings_file, 'w')
else:
    warnings_fh = sys.stderr

def warn(message):
    if warnings_fh is not None:
        warnings_fh.write("Warning: %s\n" % message)
        warnings_fh.flush()
    log(message, level=INFO)

log_fh = None
if options.log_file is not None:
    log_fh = open(options.log_file, 'w')
else:
    log_fh = sys.stderr

NONE=0
INFO=1
DEBUG=2
TRACE=3
debug_level = options.debug_level
if debug_level is None:
    debug_level = INFO
def log(message, level=INFO, end_char='\n'):
    if level <= debug_level:
        log_fh.write("%s%s" % (message, end_char))
        log_fh.flush()

#utility function to map names or indices to column indicies
def _get_col(col_name_or_index, header_cols, require_match=True):
    try:
        if col_name_or_index is None:
            raise ValueError
        return(int(col_name_or_index))
    except ValueError:
        matching_cols = [i for i in range(0,len(header_cols)) if header_cols[i] == col_name_or_index]
        if len(matching_cols) == 0:
            if require_match:
                bail("Could not find match for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
            else:
                return None
        if len(matching_cols) > 1:
            bail("Found two matches for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
        return matching_cols[0]

def _construct_map_to_ind(values):
    return dict([(values[i], i) for i in range(len(values))])

def is_gz_file(filepath, is_remote, flag=None):

    if len(filepath) >= 3 and (filepath[-3:] == ".gz" or filepath[-4:] == ".bgz") and (flag is None or 'w' not in flag):
        try:
            with gzip.open(filepath) as test_fh:
                try:
                    test_fh.readline()
                    return True
                except gzip.BadGzipFile:
                    return False
        except FileNotFoundError:
            return True

    elif flag is None or 'w' not in flag:
        flag = 'rb'
        if is_remote:
            import urllib.request
            test_f = urllib.request.urlopen(filepath)
        else:
            test_f = open(filepath, 'rb')

        is_gz = test_f.read(2) == b'\x1f\x8b'
        test_f.close()
        return is_gz
    else:
        return filepath[-3:] == ".gz" or filepath[-4:] == ".bgz"

def open_gz(file, flag=None):
    is_remote = False
    remote_prefixes = ["http:", "https:", "ftp:"]
    for remote_prefix in remote_prefixes:
        if len(file) >= len(remote_prefix) and file[:len(remote_prefix)] == remote_prefix:
            is_remote = True
       
    if is_gz_file(file, is_remote, flag=flag):
        open_fun = gzip.open
        if flag is not None and len(flag) > 0 and not flag[-1] == 't':
            flag = "%st" % flag
        elif flag is None:
            flag = "rt"
    else:
        open_fun = open

    if is_remote:
        import urllib.request
        import io
        if flag is not None:
            if open_fun is open:
                fh = io.TextIOWrapper(urllib.request.urlopen(file, flag))
            else:
                fh = open_fun(urllib.request.urlopen(file), flag)
        else:
            if open_fun is open:
                fh = io.TextIOWrapper(urllib.request.urlopen(file))
            else:
                fh = open_fun(urllib.request.urlopen(file))
    else:
        if flag is not None:
            try:
                fh = open_fun(file, flag, encoding="utf-8")
            except LookupError:
                fh = open_fun(file, flag)
        else:
            try:
                fh = open_fun(file, encoding="utf-8")
            except LookupError:
                fh = open_fun(file)

    return fh


def read_gene_or_pheno_factors(gene_or_pheno_factors_in, factor_col_names, id_col=None, combined_col=None, direct_col=None, indirect_col=None, read_phenos=False, node_full_size=None, node_min_size=None, node_full_color=None, max_num_per_factor=None, min_combined=0, min_indirect=0, keep_zero_factors=False, min_loading=0, force_include=None):

    if gene_or_pheno_factors_in is None:
        bail("Require %s for this operation" % ("--pheno-factors-in" if read_phenos else "--gene-factors-in"))

    log("Reading file %s" % gene_or_pheno_factors_in, INFO)

    if id_col is None:
        id_col = "Pheno" if read_phenos else "Gene"        

    if combined_col is None:
        combined_col = "combined"        
    if direct_col is None:
        direct_col = "log_bf"        
    if indirect_col is None:
        indirect_col = "prior"        

    factor_to_max_weight = {}
    factor_to_weight = {}
    combined_direct = {}
    factor_matrix = []
    genes_or_phenos = []

    delim = None
    with open_gz(gene_or_pheno_factors_in) as gene_or_pheno_factors_fh:
        header_line = gene_or_pheno_factors_fh.readline()
        if '\t' in header_line:
            delim = '\t'
        header_cols = header_line.strip().split(delim)

        id_col = _get_col(id_col, header_cols)

        combined_col = _get_col(combined_col, header_cols, require_match=False)
        direct_col = _get_col(direct_col, header_cols, require_match=False)
        indirect_col = None
        if indirect_col is not None:
            indirect_col = _get_col(indirect_col, header_cols, require_match=False)

        factors = [str(x) for x in factor_col_names]
        factor_cols = [_get_col(x, header_cols) for x in factor_col_names]

        for line in gene_or_pheno_factors_fh:
            cols = line.strip().split(delim)
            if id_col >= len(cols):
                warn("Skipping due to too few columns in line: %s" % line)
                continue

            gene_or_pheno = cols[id_col]

            combined = 1
            if combined_col is not None:
                combined = cols[combined_col]
                try:
                    combined = float(combined)
                except ValueError:
                    warn("Skipping unconvertible value %s for %s" % (combined, gene_or_pheno))
                    continue

            direct = 1
            if direct_col is not None:
                direct = cols[direct_col]
                try:
                    direct = float(direct)
                except ValueError:
                    warn("Skipping unconvertible value %s for %s" % (direct, gene_or_pheno))
                    continue

            indirect = combined - direct
            if indirect_col is not None:
                indirect = cols[indirect_col]
                try:
                    indirect = float(indirect)
                except ValueError:
                    warn("Skipping unconvertible value %s for %s" % (indirect, gene_or_pheno))
                    continue

            if (min_combined is not None and combined < min_combined) or (min_indirect is not None and indirect < min_indirect):
                continue

            cur_factor_to_weight = {}
            add = False
            current_values = []
            for i in range(len(factor_cols)):
                factor_col = factor_cols[i]
                if factor_col >= len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    break

                factor = factors[i]
                try:
                    weight = float(cols[factor_col])
                except ValueError:
                    if not cols[factor_col] == "NA":
                        warn("Skipping unconvertible value %s for %s" % (cols[factor_col], gene_or_pheno))
                    continue

                total_weight = weight
                cur_factor_to_weight[factor] = weight

                if (total_weight != 0 or keep_zero_factors) and weight > min_loading:
                    add = True
                if factor not in factor_to_max_weight or weight > factor_to_max_weight[factor]:
                    factor_to_max_weight[factor] = weight

                current_values.append(weight)

            if not add:
                continue

            if node_full_color is not None:
                if direct > node_full_color:
                    direct = node_full_color

                direct /= node_full_color

            factor_to_weight[gene_or_pheno] = cur_factor_to_weight
            combined_direct[gene_or_pheno] = (combined, direct)
            factor_matrix.append(current_values)
            genes_or_phenos.append(gene_or_pheno)


        #for gene_or_pheno in factor_to_weight:
        #    for factor in factor_to_weight[gene_or_pheno]:
        #        norm = np.array(factor_to_max_weight[factor])
        #        norm[norm == 0] = 1
        #        factor_to_weight[gene_or_pheno][factor] /= norm

    factor_matrix = np.array(factor_matrix)
    if max_num_per_factor is not None and len(genes_or_phenos) > max_num_per_factor:

        factor_matrix_perturbed = factor_matrix + ((np.random.random(factor_matrix.shape) - 0.5) * 1e-4)

        if max_num_per_factor > 0:

            #add random noise just in case all are tied
            factor_thresholds = np.partition(factor_matrix_perturbed, len(genes_or_phenos) - max_num_per_factor, axis=0)[-max_num_per_factor]
        else:
            factor_thresholds = np.inf


        genes_or_phenos_to_include_mask = np.any(factor_matrix_perturbed >= factor_thresholds, axis=1)

        if force_include:
            for i in range(len(genes_or_phenos)):
                if genes_or_phenos[i] in force_include:
                    genes_or_phenos_to_include_mask[i] = True

        if np.sum(~genes_or_phenos_to_include_mask) > 0:
            genes_or_phenos = [genes_or_phenos[i] for i in range(len(genes_or_phenos)) if genes_or_phenos_to_include_mask[i]]

            factor_matrix = factor_matrix[genes_or_phenos_to_include_mask,:]
            factor_to_weight = {key: factor_to_weight[key] for key in genes_or_phenos if key in factor_to_weight}
            combined_direct = {key: combined_direct[key] for key in genes_or_phenos if key in combined_direct}

    
    if node_full_size is None and len(combined_direct) > 0:
        max_combined = max([combined_direct[k][0] for k in combined_direct])
        node_full_size = max_combined


    if node_min_size is None and node_full_size is not None:
        node_min_size = 0.2 * node_full_size

    for k in combined_direct:
        if combined_direct[k][0] > node_full_size:
            combined_direct[k] = (node_full_size, combined_direct[k][1])

        if combined_direct[k][0] < node_min_size:
            combined_direct[k] = (node_min_size, combined_direct[k][1])


        if node_full_size > 0:
            combined_direct[k] = (combined_direct[k][0] / node_full_size, combined_direct[k][1])

    #if len(genes_or_phenos) == 0:
    #    bail("Empty file!")

    return factor_to_weight, combined_direct, factors, factor_matrix, genes_or_phenos

def read_gene_or_pheno_anchor_weights(gene_or_pheno_factors_in, anchor_col, relevance_col, factor_col_names, anchors, id_col=None, read_phenos=False):

    if gene_or_pheno_factors_in is None:
        bail("Require %s for this operation" % ("--pheno-anchor-factors-in" if read_phenos else "--gene-anchor-factors-in"))

    log("Reading file %s" % gene_or_pheno_factors_in, INFO)

    if id_col is None:
        id_col = "Pheno" if read_phenos else "Gene"        

    factor_to_anchor_weights = {}
    anchor_relevance = {}
    delim = None
    anchor_to_ind = dict([(anchors[i], i) for i in range(len(anchors))])
    with open_gz(gene_or_pheno_factors_in) as factors_fh:
        header_line = factors_fh.readline()
        if '\t' in header_line:
            delim = '\t'
        header_cols = header_line.strip().split(delim)

        id_col = _get_col(id_col, header_cols)
        anchor_col = _get_col(anchor_col, header_cols)
        relevance_col = _get_col(relevance_col, header_cols)
        factors = [str(x) for x in factor_col_names]
        factor_cols = [_get_col(x, header_cols) for x in factor_col_names]

        for line in factors_fh:
            cols = line.strip().split(delim)
            if id_col >= len(cols):
                warn("Skipping due to too few columns in line: %s" % line)
                continue

            gene_or_pheno = cols[id_col]
            anchor = cols[anchor_col]
            try:
                relevance = float(cols[relevance_col])
            except ValueError:
                if not cols[factor_col] == "NA":
                    warn("Skipping unconvertible value %s for factor %s" % (cols[relevance_col], factor))
                continue

            if gene_or_pheno not in anchor_relevance:
                anchor_relevance[gene_or_pheno] = np.zeros(len(anchors))
            anchor_relevance[gene_or_pheno][anchor_to_ind[anchor]] = relevance

            cur_factor_to_anchor_weight = {}
            current_values = []
            if gene_or_pheno not in factor_to_anchor_weights:
                factor_to_anchor_weights[gene_or_pheno] = {}

            for i in range(len(factor_cols)):
                factor_col = factor_cols[i]
                if factor_col >= len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    break

                factor = factors[i]
                try:
                    weight = float(cols[factor_col])
                except ValueError:
                    if not cols[factor_col] == "NA":
                        warn("Skipping unconvertible value %s for %s" % (cols[factor_col], gene_or_pheno))
                    continue

                if factor not in factor_to_anchor_weights[gene_or_pheno]:
                    factor_to_anchor_weights[gene_or_pheno][factor] = np.zeros(len(anchors))
                factor_to_anchor_weights[gene_or_pheno][factor][anchor_to_ind[anchor]] = weight

    return anchor_relevance, factor_to_anchor_weights


def read_factors(factors_in, factors_id_col=None, factors_label_col=None, factors_beta_col=None, mech_node_full_size=None, mech_node_min_size=None):

    if factors_in is None:
        bail("Require --factors-in for this operation")

    log("Reading --factors-in file %s" % factors_in, INFO)

    if factors_id_col is None:
        factors_id_col = "Factor"
    if factors_label_col is None:
        factors_label_col = "label"
    if factors_beta_col is None:
        factors_beta_col = "relevance"

    factors = []
    factor_to_label = {}
    factor_to_beta = {}

    delim = None
    with open_gz(factors_in) as factors_fh:
        header_line = factors_fh.readline()
        if '\t' in header_line:
            delim = '\t'
        header_cols = header_line.strip().split(delim)

        factor_id_col = _get_col(factors_id_col, header_cols)
        factor_label_col = _get_col(factors_label_col, header_cols)
        factor_beta_col = _get_col(factors_beta_col, header_cols, require_match=False)
        if factor_beta_col is None:
            warn("Couldn't find column %s; trying any_relevance" % factors_beta_col)
            factors_beta_col = "any_relevance"            
            factor_beta_col = _get_col(factors_beta_col, header_cols)

        for line in factors_fh:
            cols = line.strip().split(delim)
            if factor_id_col >= len(cols):
                warn("Skipping due to too few columns in line: %s" % line)
                continue

            factor = cols[factor_id_col]
            label = cols[factor_label_col]

            beta = 0
            try:
                beta = float(cols[factor_beta_col])
            except ValueError:
                if not cols[factor_col] == "NA":
                    warn("Skipping unconvertible value %s for gene_set %s" % (cols[factor_col], gene_set))
                continue

            factors.append(factor)
            factor_to_label[factor] = label
            factor_to_beta[factor] = beta

    if mech_node_full_size is None:
        mech_node_full_size = max(factor_to_beta.values())

    if mech_node_min_size is None:
        mech_node_min_size = 0.3 * mech_node_full_size


    for factor in factor_to_beta:
        if factor_to_beta[factor] > mech_node_full_size:
            factor_to_beta[factor] = mech_node_full_size
        if factor_to_beta[factor] < mech_node_min_size:
            factor_to_beta[factor] = mech_node_min_size


        factor_to_beta[factor] /= mech_node_full_size

    return factor_to_label, factor_to_beta, factors

def read_factors_anchor_weights(factors_in, factors_anchor_col, factors_relevance_col, factors_id_col=None):

    if factors_in is None:
        bail("Require --factors-anchor-in for this operation")

    log("Reading --factors-anchor-in file %s" % factors_in, INFO)

    if factors_id_col is None:
        factors_id_col = "Factor"

    anchors = set()
    delim = None
    with open_gz(factors_in) as factors_fh:
        header_line = factors_fh.readline()
        if '\t' in header_line:
            delim = '\t'
        header_cols = header_line.strip().split(delim)

        factor_id_col = _get_col(factors_id_col, header_cols)
        factor_anchor_col = _get_col(factors_anchor_col, header_cols)

        for line in factors_fh:
            cols = line.strip().split(delim)
            if factor_id_col >= len(cols):
                warn("Skipping due to too few columns in line: %s" % line)
                continue

            factor = cols[factor_id_col]
            anchor = cols[factor_anchor_col]

            anchors.add(anchor)
    anchors = list(anchors)
    anchor_to_ind = dict([(anchors[i], i) for i in range(len(anchors))])

    factor_to_anchor_weights = {}
    factor_to_anchor_relevance = {}
    with open_gz(factors_in) as factors_fh:
        header_line = factors_fh.readline()
        if '\t' in header_line:
            delim = '\t'
        header_cols = header_line.strip().split(delim)

        factor_id_col = _get_col(factors_id_col, header_cols)
        factor_anchor_col = _get_col(factors_anchor_col, header_cols)
        factor_relevance_col = _get_col(factors_relevance_col, header_cols)

        for line in factors_fh:
            cols = line.strip().split(delim)
            if factor_id_col >= len(cols):
                warn("Skipping due to too few columns in line: %s" % line)
                continue

            factor = cols[factor_id_col]
            try:
                relevance = float(cols[factor_relevance_col])
            except ValueError:
                if not cols[factor_col] == "NA":
                    warn("Skipping unconvertible value %s for factor %s" % (cols[factor_relevance_col], factor))
                continue

            anchor = cols[factor_anchor_col]

            if factor not in factor_to_anchor_weights:
                factor_to_anchor_weights[factor] = np.zeros(len(anchors))

            if anchor not in anchor_to_ind:
                bail("Bad anchor %s" % anchor)
                
            if factor not in factor_to_anchor_relevance:
                factor_to_anchor_relevance[factor] = np.zeros(len(anchors))
            factor_to_anchor_relevance[factor][anchor_to_ind[anchor]] = relevance

    return anchors, factor_to_anchor_relevance


def read_gene_phenos(gene_pheno_in, gene_pheno_gene_col=None, gene_pheno_pheno_col=None, gene_pheno_weight_col=None, gene_pheno_edge_full_width=5, min_gene_pheno_loading=1):

    if gene_pheno_in is None:
        bail("Require --gene-pheno-in for this operation")

    log("Reading --gene-pheno-in file %s" % gene_pheno_in, INFO)

    if gene_pheno_gene_col is None:
        gene_pheno_gene_col = "Gene"
    if gene_pheno_pheno_col is None:
        gene_pheno_pheno_col = "Pheno"
    if gene_pheno_weight_col is None:
        gene_pheno_weight_col = "combined"

    gene_pheno_to_weight = {}

    delim = None
    max_weight = -np.inf
    with open_gz(gene_pheno_in) as gene_pheno_fh:
        header_line = gene_pheno_fh.readline()
        if '\t' in header_line:
            delim = '\t'
        header_cols = header_line.strip().split(delim)

        gene_col = _get_col(gene_pheno_gene_col, header_cols)
        pheno_col = _get_col(gene_pheno_pheno_col, header_cols)
        weight_col = _get_col(gene_pheno_weight_col, header_cols, require_match=False)

        for line in gene_pheno_fh:
            cols = line.strip().split(delim)
            if gene_col >= len(cols) or pheno_col > len(cols) or (weight_col is not None and weight_col > len(cols)):
                warn("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[gene_col]
            pheno = cols[pheno_col]

            weight = 1
            if weight_col is not None:
                try:
                    weight = float(cols[weight_col])
                except ValueError:
                    if not cols[weight_col] == "NA":
                        warn("Skipping unconvertible value %s for gene %s and pheno %s" % (cols[weight_col], gene, pheno))
                    continue

                if weight < min_gene_pheno_loading:
                    continue

            if gene not in gene_pheno_to_weight:
                gene_pheno_to_weight[gene] = {}
            gene_pheno_to_weight[gene][pheno] = weight
            max_weight = np.maximum(max_weight, weight)

    if gene_pheno_edge_full_width is None:
        gene_pheno_edge_full_width = max_weight

    for gene in gene_pheno_to_weight:
        for pheno in gene_pheno_to_weight[gene]:
            if gene_pheno_to_weight[gene][pheno] > gene_pheno_edge_full_width:
                gene_pheno_to_weight[gene][pheno] = gene_pheno_edge_full_width

            gene_pheno_to_weight[gene][pheno] /= gene_pheno_edge_full_width

    return gene_pheno_to_weight


def generate_distinct_colors_orig(N):

    if N == 2:
        colors = [(1,0,0), (0,0,1)]
    else:

        colors = []
        step = 256 / N  # Step to space colors evenly in the RGB space

        white_scale = 0.5

        for i in range(1,N+1):
            r = 1 - white_scale * (1 - int((i * step) % 256) / 256.0)        # Red channel
            g = 1 - white_scale * (1 - int((i * step * 2) % 256) / 256.0)    # Green channel
            b = 1 - white_scale * (1 - int((i * step * 3) % 256) / 256.0)    # Blue channel
            colors.append((r, g, b))

    return colors


def generate_distinct_colors(N, start_with_red_blue=True):
    """
    Generate N distinct colors, ensuring complementarity if starting with red and blue.
    Defaults to the original behavior if not starting with red and blue.

    Args:
        N (int): The number of distinct colors to generate.
        start_with_red_blue (bool): If True, the first two colors are red and blue.

    Returns:
        list: A list of tuples representing RGB colors.
    """

    if N <= 0:
        return []

    colors = []

    if start_with_red_blue and N >= 2:
        # Start with predefined red and blue
        colors.append((1, 0, 0))  # Red
        colors.append((0, 0, 1))  # Blue

        import colorsys

        # Generate the remaining colors with complementarity
        for i in range(2, N):
            max_dist_color = None
            max_dist = -1

            # Sample potential colors in HSV space and find the most distinct one
            for h in range(0, 360, 10):  # Test hues in 10-degree increments
                for s in [0.7, 1.0]:     # Test two saturation levels
                    for v in [0.7, 1.0]: # Test two brightness levels
                        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
                        candidate_color = (r, g, b)

                        # Compute the minimum Euclidean distance to all existing colors
                        min_dist = min(
                            sum((r1 - r2)**2 for r1, r2 in zip(candidate_color, c))**0.5
                            for c in colors
                        )

                        # Keep track of the most distinct candidate
                        if min_dist > max_dist:
                            max_dist = min_dist
                            max_dist_color = candidate_color

            # Add the most distinct color to the list
            colors.append(max_dist_color)

    else:
        # Default behavior: evenly distribute colors in RGB space
        step = 256 / N  # Step to space colors evenly
        white_scale = 0.5

        for i in range(1, N + 1):
            r = 1 - white_scale * (1 - int((i * step) % 256) / 256.0)  # Red channel
            g = 1 - white_scale * (1 - int((i * step * 2) % 256) / 256.0)  # Green channel
            b = 1 - white_scale * (1 - int((i * step * 3) % 256) / 256.0)  # Blue channel
            colors.append((r, g, b))

    return colors


# Function to blend colors based on weights
def blend_colors(color_list, weights, opacity=1):
    weights = weights / np.sum(weights)

    if opacity < 0.1:
        opacity = 0.1
    if opacity > 1:
        opacity = 1


    color_list = np.array(color_list)

    blended_color = 1 - opacity * np.average(1 - color_list, axis=0, weights=weights)

    #blended_color = np.average(color_list, axis=0, weights=weights)



    blended_color[blended_color < 1/256.0] = 0
    blended_color[blended_color > 1] = 1


    return tuple(c for c in blended_color)  # Convert to 0-255 for HTML colors

def rgb_to_hex(rgb, alpha=1):

    value = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    if alpha < 1 and alpha >= 0:
        value += "{:02x}".format(int(alpha * 255))
    return value

# Helper function to add nodes and edges to a NetworkX graph
def add_nodes_and_edges(G, genes, gene_factor_to_weight, gene_to_combined_direct, gene_factors, phenos, pheno_factor_to_weight, pheno_to_combined_direct, pheno_factors, gene_sets, gene_set_to_label, gene_set_factor_to_weight, gene_set_to_beta, gene_set_factors, gene_set_to_genes, gene_set_to_phenos, colors, gene_pheno_to_weight=None, node_size_scale=1, gene_node_size_further_scale=1, pheno_node_size_further_scale=1, beta_node_size_further_scale=5, edge_max_width=3, gene_node_opacity=1, pheno_node_opacity=0.5, edge_opacity=0.5, factor_to_anchor_relevance=None, pheno_factor_to_anchor_weights=None, gene_factor_to_anchor_weights=None, pheno_factor_to_anchor_relevance=None, gene_factor_to_anchor_relevance=None):
    log("Adding nodes and edges")
    # Add nodes from the first file
    node_size_scale *= 10000

    graph_nodes = set()

    if gene_factor_to_anchor_relevance is not None:
        max_gene_factor_relevance = max([np.max(gene_factor_to_anchor_relevance[node]) for node in genes])
    else:
        max_gene_factor_relevance = 1

    for node in genes:
        size = gene_to_combined_direct[node][0] * node_size_scale * gene_node_size_further_scale  # Node size

        if gene_factor_to_anchor_relevance is not None:
            color_weights = gene_factor_to_anchor_relevance[node]
            opacity = np.max(color_weights) / max_gene_factor_relevance
        else:
            opacity = gene_to_combined_direct[node][1]
            color_weights = np.array([gene_factor_to_weight[node][x] for x in gene_factors])

        node_color = blend_colors(colors, color_weights, opacity=opacity)
        node_border_color = blend_colors(colors, color_weights, opacity=1)
        # Add node with its properties
        graph_nodes.add(node)
        G.add_node(node, size=size, color=node_color, border_color=node_border_color, alpha=gene_node_opacity, label=node, gene=True)

    if pheno_factor_to_anchor_relevance is not None:
        max_pheno_factor_relevance = max([np.max(pheno_factor_to_anchor_relevance[node]) for node in phenos])
    else:
        max_pheno_factor_relevance = 1

    pheno_to_color = {}
    if phenos is not None:
        for node in phenos:
            size = pheno_to_combined_direct[node][0] * node_size_scale * pheno_node_size_further_scale  # Node size
            if pheno_factor_to_anchor_relevance is not None:
                color_weights = pheno_factor_to_anchor_relevance[node]
                opacity = np.max(color_weights) / max_pheno_factor_relevance
            else:
                opacity = pheno_to_combined_direct[node][1]
                color_weights = np.array([pheno_factor_to_weight[node][x] for x in pheno_factors])

            node_color = blend_colors(colors, color_weights, opacity=opacity)
            node_border_color = blend_colors(colors, color_weights, opacity=1)
            pheno_to_color[node] = node_border_color

            # Add node with its properties
            graph_nodes.add(node)
            G.add_node(node, size=size, color=node_color, border_color=node_border_color, alpha=pheno_node_opacity, label=node, shape='triangle', pheno=True)

    # Create bipartite connections
    if factor_to_anchor_relevance is not None:
        max_factor_relevance = max([np.max(factor_to_anchor_relevance[node_set]) for node_set in gene_sets])
    else:
        max_factor_relevance = 1

    for node_set in gene_sets:
        if node_set not in gene_set_to_genes and node_set not in gene_set_to_phenos:
            continue

        nodes = {}
        if node_set in gene_set_to_genes:
            nodes.update(gene_set_to_genes[node_set])
        if node_set in gene_set_to_phenos:
            nodes.update(gene_set_to_phenos[node_set])
        
        # Get the color and opacity of the node set
        set_size = gene_set_to_beta[node_set] * node_size_scale

        if factor_to_anchor_relevance is not None:
            set_color_weights = factor_to_anchor_relevance[node_set]
            set_opacity = np.max(set_color_weights) / max_factor_relevance
        else:
            set_color_weights = np.array([gene_set_factor_to_weight[node_set][x] for x in gene_set_factors])
            set_opacity = 1

        node_set_color = blend_colors(colors, set_color_weights, opacity=set_opacity)
        node_set_border_color = blend_colors(colors, set_color_weights, opacity=1)
        

        # Add a new node for the node set (with a special shape or size)

        graph_nodes.add(node_set)
        G.add_node(node_set, size=set_size * beta_node_size_further_scale, color=node_set_color, border_color=node_set_border_color, alpha=1, label=gene_set_to_label[node_set], shape='square', bipartite=True)

        # Connect all nodes in the set to the new node representing the node set
        for node in nodes:
            node_weight_in_set = nodes[node]

            if 'pheno' in G.nodes[node]:
                if pheno_factor_to_anchor_weights is not None:
                    color_weights = pheno_factor_to_anchor_weights[node][node_set]
                    node_set_edge_color = blend_colors(colors, color_weights, opacity=set_opacity)
                else:
                    node_set_edge_color = node_set_color
            else:
                if gene_factor_to_anchor_weights is not None:
                    color_weights = gene_factor_to_anchor_weights[node][node_set]
                    node_set_edge_color = blend_colors(colors, color_weights, opacity=set_opacity)
                else:
                    node_set_edge_color = node_set_color

            G.add_edge(node_set, node, width=edge_max_width * node_weight_in_set, color=node_set_edge_color, alpha=edge_opacity, dashed=False)

    if gene_pheno_to_weight is not None:
        for gene in gene_pheno_to_weight:
            if gene not in graph_nodes:
                continue
            for pheno in gene_pheno_to_weight[gene]:
                if pheno not in graph_nodes:
                    continue

                edge_color = None
                if pheno in pheno_to_color:
                    edge_color = pheno_to_color[pheno]

                weight = gene_pheno_to_weight[gene][pheno]

                G.add_edge(gene, pheno, width=edge_max_width * weight, color=edge_color, alpha=edge_opacity, dashed=True)


# Helper function to save graph as PDF
def save_pdf(G, pos, output_file, colors, color_labels, node_label_size_threshold=1000, node_border_width=4, coordinate_scale=1, pdf_width=50, pdf_height=50, font_size=14):
    log("Saving %s" % output_file)

    plt.figure(figsize=(pdf_width, pdf_height))
    
    normal_nodes = [n for n, d in G.nodes(data=True) if not d.get("bipartite", False)]
    is_pheno_node = [d.get("pheno", False) for n, d in G.nodes(data=True) if not d.get("bipartite", False)]
    bipartite_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite", False)]
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in normal_nodes]
    node_border_colors = [G.nodes[node]['border_color'] for node in normal_nodes]
    node_sizes = [G.nodes[normal_nodes[i]]['size'] / coordinate_scale * (0.5 if is_pheno_node[i] else 1) for i in range(len(normal_nodes))]
    node_opacities = np.array([G.nodes[node]['alpha'] for node in normal_nodes])
    node_shapes = ["^" if x else "o" for x in is_pheno_node]

    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color=[rgb_to_hex(x) for x in node_colors], edgecolors=[rgb_to_hex(x) for x in node_border_colors], linewidths=node_border_width, node_size=node_sizes, alpha=node_opacities)

    #Draw node set nodes
    bipartite_node_colors = [G.nodes[node]['color'] for node in bipartite_nodes]
    bipartite_node_sizes = [G.nodes[node]['size'] / coordinate_scale for node in bipartite_nodes]
    bipartite_node_opacities = [G.nodes[node]['alpha'] for node in bipartite_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=bipartite_nodes, node_color=[rgb_to_hex(x) for x in bipartite_node_colors], node_size=bipartite_node_sizes, node_shape="s", alpha=bipartite_node_opacities)


    # Draw edges
    edge_colors = [G.edges[edge]['color'] for edge in G.edges()]
    edge_opacities = np.array([G.edges[edge]['alpha'] for edge in G.edges()])
    edge_widths = np.array([G.edges[edge]['width'] for edge in G.edges()])
    edge_styles = ["-" if G.edges[edge]['dashed'] else "solid" for edge in G.edges()]

    edge_opacities = 1
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color=[rgb_to_hex(x) for x in edge_colors], style=edge_styles, width=edge_widths, alpha=edge_opacities)

    # Draw labels for nodes above a size threshold
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes() if G.nodes[node]['size'] > node_label_size_threshold}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size)

    # Create a legend for the 10 colors and their labels
    color_legend = []
    assert(len(colors) == len(color_labels))
    for i in range(len(colors)):
        color_legend.append(mpatches.Patch(color=colors[i], label=color_labels[i]))  # Adjusting for 0-based index

    # Add the legend to the plot
    plt.legend(handles=color_legend, title="Color Legend")

    # Save the plot as a PDF file with custom width and height
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()

# Helper function to generate interactive HTML with Pyvis
def save_html(G, pos, output_file, colors, color_labels, html_pos_scale=200, html_edge_width_scale=0.4, html_node_size_scale=0.02, node_border_width=4, html_node_border_scale=0.75, html_height=1000, font_size=14, use_physics=False, add_links=False):

    from pyvis.network import Network

    log("Saving %s" % output_file)
    net = Network(notebook=False, width="100%", height="%spx" % html_height)

    # Add nodes and edges from the NetworkX graph to Pyvis
    #for node in G.nodes(data=True):

    num_pheno_nodes = len([node for node in pos if G.nodes[node].get('pheno')])

    add_phenos = num_pheno_nodes > 0
    add_genes = len(pos) - num_pheno_nodes > 0

    for node, position in pos.items():
        node_id = node
        node_data = G.nodes[node]
        shape = 'square' if node_data.get('bipartite', False) else 'dot'
        size = node_data.get('size') * html_node_size_scale
        color = rgb_to_hex(node_data.get('color'))
        #shape = 'icon'
        #triangle_icon = {"face": "FontAwesome", "code": "\uf0d8", "size": size, "color": color}

        #border_width = 0 if node_data.get('bipartite', False) else node_border_width * html_node_border_scale
        border_width = node_border_width * html_node_border_scale
        if add_phenos and node_data.get('pheno'):
            size *= 0.5
            entity = "phenotype"
        else:
            entity = "gene"
        

        output_root = os.path.splitext(os.path.basename(output_file))[0]
        output_root = output_root.replace(".html", "")

        url="#"
        if add_links:
            if node_data.get('bipartite', False):
                url = "%s.%s.html" % (output_root, node_id)
            elif not node_data.get('pheno', False):
                url = "https://a2f.hugeamp.org:8000/pigean/%s.html?%s=%s&genesetSize=small&sigma=sigma2" % (entity, entity, node_id)

        font_color="gray" if add_phenos and add_genes and node_data.get('pheno') else "black"

        net.add_node(
            node_id,
            label=node_data.get('label'),
            size=size,  # Scale down size for better viewing
            color=color,
            border=rgb_to_hex(node_data.get('color')),
            border_color=rgb_to_hex(node_data.get('border_color')),
            borderWidth = border_width,
            opacity=node_data.get('alpha'),
            x=position[0] * html_pos_scale,
            y=-position[1] * html_pos_scale,
            font={'size': font_size, 'color': font_color},
            url=url,
            shape=shape
        )

    for edge in G.edges(data=True):
        source, target, edge_data = edge
        net.add_edge(
            source,
            target,
            color=rgb_to_hex(edge_data.get('color'), alpha=edge_data.get('alpha')),
            dashes=edge_data.get('dashed'),
            width=edge_data.get('width') * html_edge_width_scale,
        )

    if use_physics:

        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 10
            },
            "timestep": 0.5,
            "adaptiveTimestep": true,
            "barnesHut": {
              "gravitationalConstant": -10000,
              "centralGravity": 0.5,
              "springLength": 100,
              "springConstant": 0.05,
              "damping": 0.5
            },
            "minVelocity": 0.1,
            "solver": "barnesHut"
          },
          "interaction": {
            "dragNodes": true,
            "hideEdgesOnDrag": false,
            "physicsAfterStabilization": true
          },
          "manipulation": {
            "enabled": false
          }
        }
        """)

    else:
        net.toggle_physics(False)


    # Create a legend for the colors and their labels
    color_legend = {}
    assert(len(colors) == len(color_labels))
    for i in range(len(colors)):
        color_legend[color_labels[i]] = 'rgb{}'.format(colors[i])  # Adjusting for 0-based index

    # Save the network as an HTML file for interactive viewing
    try:
        if os.path.isfile(output_file):
            os.remove(output_file)
        net.write_html(output_file)
    except FileNotFoundError:
        if not os.path.isfile(output_file):
            raise

    
    if add_links:

        # Inject JavaScript to handle node click and open the link
        javascript = """
            <script type="text/javascript">
                network.on("click", function(params) {
                    if (params.nodes.length === 1) {  // Check if a single node was clicked
                        var clickedNodeId = params.nodes[0];
                        var nodeData = network.body.data.nodes.get(clickedNodeId);
                        if (nodeData.url) {
                            window.open(nodeData.url, '_blank');  // Open the URL in a new tab
                        }
                    }
                });
            </script>
        """

        # Inject the JavaScript code into the HTML file
        with open(output_file, "r") as file:
            html_content = file.read()

        # Inject the JavaScript just before the closing </body> tag
        html_content = html_content.replace("</body>", javascript + "</body>")

        # Save the modified HTML file
        with open(output_file, "w") as file:
            file.write(html_content)


def save_json(G, pos, output_file, html_pos_scale=200, html_edge_width_scale=0.4, html_node_size_scale=0.02, node_border_width=4, html_node_border_scale=0.75, html_height=1000, font_size=14):
    """
    Save nodes and edges of a NetworkX graph to a JSON file in the specified format.
    
    Parameters:
    - G (networkx.Graph): The graph with nodes and edges.
    - output_file (str): The filename for the JSON output.
    """

    import json

    log("Saving %s" % output_file)

    # Add nodes and edges from the NetworkX graph to Pyvis
    #for node in G.nodes(data=True):

    num_pheno_nodes = len([node for node in pos if G.nodes[node].get('pheno')])

    add_phenos = num_pheno_nodes > 0
    add_genes = len(pos) - num_pheno_nodes > 0

    nodes = []

    for node, position in pos.items():
        node_id = node
        node_data = G.nodes[node]
        shape = 'square' if node_data.get('bipartite', False) else 'dot'
        size = node_data.get('size') * html_node_size_scale
        color = rgb_to_hex(node_data.get('color'))
        #shape = 'icon'
        #triangle_icon = {"face": "FontAwesome", "code": "\uf0d8", "size": size, "color": color}

        #border_width = 0 if node_data.get('bipartite', False) else node_border_width * html_node_border_scale
        border_width = node_border_width * html_node_border_scale
        if add_phenos and node_data.get('pheno'):
            size *= 0.5
            entity = "phenotype"
        else:
            entity = "gene"
        
        font_color="gray" if add_phenos and add_genes and node_data.get('pheno') else "black"
        node_info = {
                "border": rgb_to_hex(node_data.get('color')),
                "borderWidth": border_width,
                "border_color": rgb_to_hex(node_data.get('border_color')),
                "color": color,
                "font": {'size': font_size, 'color': font_color},
                "id": node_id,
                "label": node_data.get('label'),
                "opacity": node_data.get('alpha'),
                "shape": shape,
                "size": size,
                "x": position[0] * html_pos_scale,
                "y": position[1] * html_pos_scale
            }
        nodes.append(node_info)

    edges = []
    for edge in G.edges(data=True):
        source, target, edge_data = edge
        edge_info = {
            "color": rgb_to_hex(edge_data.get('color'), alpha=edge_data.get('alpha')),
            "dashes": edge_data.get('dashed'),
            "from": source,
            "to": target,
            "width": edge_data.get('width') * html_edge_width_scale
        }
        edges.append(edge_info)

    data = {
            "nodes": nodes,
            "edges": edges
        }

    # Write to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def weighted_mds(X, weights=1, n_components=2):
    """
    Perform Weighted Classical MDS on a given distance matrix D.
    
    Parameters:
    - D: Distance matrix (NxN, symmetric).
    - weights: Weight matrix (NxN), specifying the importance of preserving distances.
    - n_components: Number of dimensions for the output (default is 2).
    
    Returns:
    - X: Configuration matrix (Nx2) with coordinates in the lower-dimensional space.
    """
    D = compute_distance_matrix(X)

    N = len(D)
    
    # Step 1: Apply the weight matrix to the distance matrix
    D_weighted = D * weights
    
    # Step 2: Centering matrix
    H = np.eye(N) - np.ones((N, N)) / N
    
    # Step 3: Double center the weighted distance matrix
    D_squared_weighted = D_weighted ** 2
    B = -0.5 * H.dot(D_squared_weighted).dot(H)
    
    # Step 4: Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Step 5: Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Step 6: Select top n_components eigenvalues and eigenvectors
    L = np.diag(np.sqrt(eigvals[:n_components]))
    V = eigvecs[:, :n_components]
    
    # Step 7: Compute the final embedding
    X = V.dot(L)
    
    return X

def compute_distance_matrix(X, min_distance=0.01, max_distance=2):
    
    if X.shape[1] == 1:
        D = np.abs(X - X[:,0])
        return(D)

    correlation_matrix = np.corrcoef(X + (np.random.random(X.shape) - 0.5) * 1e-5)
    D = 1 - correlation_matrix

    #else:
    #    X_square = np.sum(X**2, axis=1)
    #    D = (X_square[:, np.newaxis] + X_square[np.newaxis, :] - 2 * np.dot(X, X.T))
    #    D[D < 0] = 0
    #    D = np.sqrt(D)

    #replace relationships between nodes and node sets
    num_features = X.shape[1]
    num_nodes = X.shape[0] - X.shape[1]

    #replace within node set distances
    correlation_matrix2 = np.corrcoef(X[:num_nodes,:], rowvar=False)
    D[num_nodes:,num_nodes:] = 1 - correlation_matrix2

    #now node to node set distances

    #deprecated distances
    #distance from max across factor
    #max_based_distances = 1 - X / np.max(X[:num_nodes,:], axis=0)
    #distance from max across gene
    #max_based_distances = 1 - (X[:num_nodes,:].T / np.max(X[:num_nodes,:], axis=1)).T
    #distance from max across all factors and genes
    #max_based_distances = 1 - X / np.max(X[:num_nodes,:])

    #weighted sum
    max_based_distances = (X.T / np.sum(X, axis=1)).T.dot(D[num_nodes:,num_nodes:])

    #num_nodes x num_features
    D[:num_nodes,num_nodes:] = max_based_distances[:num_nodes,:]


    #adjust boundary cases

    D[D < min_distance] = min_distance
    D[D > max_distance] = max_distance
    np.fill_diagonal(D, 0)

    return(D)


# Main logic to read data and generate graphs
def main():
    # Read the TSV files
    if options.gene_factors_in is None and options.pheno_factors_in is None:
        bail("Need --gene-factors-in")
    if options.factors_in is None:
        bail("Need --factors-in")
        
    add_phenos = False
    if options.pheno_factors_in is not None:
        add_phenos = True
    add_genes = False
    if options.gene_factors_in is not None:
        add_genes = True

    if options.gene_node_full_color is None:
        if add_phenos:
            options.gene_node_full_color = 0.5
        else:
            options.gene_node_full_color = 4

   
    factor_to_label, factor_to_beta, factors = read_factors(options.factors_in, factors_id_col=options.factors_id_col, factors_label_col=options.factors_label_col, mech_node_full_size=options.mech_node_full_size, mech_node_min_size=options.mech_node_min_size)

    anchors = None
    factor_to_anchor_relevance = None
    if options.factors_anchor_in is not None:
        anchors, factor_to_anchor_relevance = read_factors_anchor_weights(options.factors_anchor_in, factors_anchor_col=options.factors_anchor_col, factors_relevance_col=options.factors_relevance_col, factors_id_col=options.factors_id_col)

    pheno_factor_to_weight = None
    pheno_to_combined_direct = None
    pheno_factors = None
    pheno_factor_matrix = None
    phenos = []
    gene_pheno_to_weight = None
    pheno_factor_to_anchor_weights = None
    pheno_factor_to_anchor_relevance = None

    if add_phenos:
        for pheno_factors_in in options.pheno_factors_in:
            cur_pheno_factor_to_weight, cur_pheno_to_combined_direct, cur_pheno_factors, cur_pheno_factor_matrix, cur_phenos = read_gene_or_pheno_factors(pheno_factors_in, factors, read_phenos=True, id_col=options.pheno_factors_id_col, combined_col=options.pheno_factors_combined_col, direct_col=options.pheno_factors_direct_col, indirect_col=options.pheno_factors_indirect_col, node_full_size=options.pheno_node_full_size, node_min_size=options.pheno_node_min_size, node_full_color=options.pheno_node_full_color, max_num_per_factor=options.max_num_pheno_nodes_per_factor, min_combined=options.pheno_min_combined, min_indirect=options.pheno_min_indirect, min_loading=options.pheno_min_loading, force_include=options.include_phenos)
            if cur_phenos is None or len(cur_phenos) == 0:
                continue
            if pheno_factor_to_weight is None:
                pheno_factor_to_weight, pheno_to_combined_direct, pheno_factors, pheno_factor_matrix, phenos = cur_pheno_factor_to_weight, cur_pheno_to_combined_direct, cur_pheno_factors, cur_pheno_factor_matrix, cur_phenos

            else:
                if len(pheno_factors) != len(cur_pheno_factors):
                    bail("Multiple --pheno-factors-in must have same number of factors")
                phenos += cur_phenos
                pheno_factor_to_weight.update(cur_pheno_factor_to_weight)
                pheno_to_combined_direct.update(cur_pheno_to_combined_direct)

                pheno_factor_matrix = np.vstack((pheno_factor_matrix, cur_pheno_factor_matrix))
                
        if options.pheno_anchor_factors_in:
            pheno_factor_to_anchor_weights = {}
            pheno_factor_to_anchor_relevance = {}
            for anchor_pheno_factors_in in options.pheno_anchor_factors_in:
                cur_pheno_factor_to_anchor_relevance, cur_pheno_factor_to_anchor_weights = read_gene_or_pheno_anchor_weights(anchor_pheno_factors_in, anchor_col=options.pheno_anchor_factors_anchor_col, relevance_col=options.pheno_anchor_factors_relevance_col, factor_col_names=factors, anchors=anchors, id_col=options.pheno_factors_id_col, read_phenos=True)
                if pheno_factor_to_anchor_weights is None:
                    pheno_factor_to_anchor_weights = cur_pheno_factor_to_anchor_weights
                    pheno_factor_to_anchor_relevance = cur_pheno_factor_to_anchor_relevance
                else:
                    pheno_factor_to_anchor_weights.update(cur_pheno_factor_to_anchor_weights)
                    pheno_factor_to_anchor_relevance.update(cur_pheno_factor_to_anchor_relevance)

        if options.gene_pheno_in is not None:
            gene_pheno_to_weight = read_gene_phenos(options.gene_pheno_in, gene_pheno_gene_col=options.gene_pheno_gene_col, gene_pheno_pheno_col=options.gene_pheno_pheno_col, gene_pheno_weight_col=options.gene_pheno_weight_col, gene_pheno_edge_full_width=options.gene_pheno_edge_full_width)
        if len(phenos) == 0:
            bail("No phenos!")

    gene_factor_to_weight = {}
    gene_to_combined_direct = None
    gene_factors = None
    gene_factor_matrix = None
    genes = []
    gene_factor_to_anchor_weights = None
    gene_factor_to_anchor_relevance = None
    if add_genes:

        for gene_factors_in in options.gene_factors_in:
            cur_gene_factor_to_weight, cur_gene_to_combined_direct, cur_gene_factors, cur_gene_factor_matrix, cur_genes = read_gene_or_pheno_factors(gene_factors_in, factors, id_col=options.gene_factors_id_col, combined_col=options.gene_factors_combined_col, direct_col=options.gene_factors_direct_col, indirect_col=options.gene_factors_indirect_col, read_phenos=False, node_full_size=options.gene_node_full_size, node_min_size=options.gene_node_min_size, node_full_color=options.gene_node_full_color, max_num_per_factor=options.max_num_gene_nodes_per_factor, min_combined=options.gene_min_combined, min_indirect=options.gene_min_indirect, min_loading=options.gene_min_loading, force_include=options.include_genes)

            if cur_genes is None or len(cur_genes) is None:
                continue
            if gene_to_combined_direct is None:
                gene_factor_to_weight, gene_to_combined_direct, gene_factors, gene_factor_matrix, genes = cur_gene_factor_to_weight, cur_gene_to_combined_direct, cur_gene_factors, cur_gene_factor_matrix, cur_genes
            else:
                if len(gene_factors) != len(cur_gene_factors):
                    bail("Multiple --gene-factors-in must have same number of factors")
                genes += cur_genes
                gene_factor_to_weight.update(cur_gene_factor_to_weight)
                gene_to_combined_direct.update(cur_gene_to_combined_direct)
                gene_factor_matrix = np.vstack((gene_factor_matrix, cur_gene_factor_matrix))
        if len(genes) == 0:
            bail("No genes!")

        if options.gene_anchor_factors_in:
            gene_factor_to_anchor_relevance = {}
            gene_factor_to_anchor_weights = {}
            for anchor_gene_factors_in in options.gene_anchor_factors_in:
                cur_gene_factor_to_anchor_relevance, cur_gene_factor_to_anchor_weights = read_gene_or_pheno_anchor_weights(anchor_gene_factors_in, anchor_col=options.gene_anchor_factors_anchor_col, relevance_col=options.gene_anchor_factors_relevance_col, factor_col_names=factors, anchors=anchors, id_col=options.gene_factors_id_col, read_phenos=False)
                if gene_factor_to_anchor_weights is None:
                    gene_factor_to_anchor_weights = cur_gene_factor_to_anchor_weights
                    gene_factor_to_anchor_relevance = cur_gene_factor_to_anchor_relevance
                else:
                    gene_factor_to_anchor_weights.update(cur_gene_factor_to_anchor_weights)
                    gene_factor_to_anchor_relevance.update(cur_gene_factor_to_anchor_relevance)

    #use the factors for gene sets instead

    gene_set_factor_to_weight = {}
    gene_set_to_beta = {}
    gene_set_factors = []
    gene_set_factor_matrix = []
    gene_set_to_label = {}
    gene_sets = []
    for factor_int1 in factors:
        factor = factor_int1
        gene_set_to_label[factor] = factor_to_label[factor_int1]
        gene_set_factor_to_weight[factor] = {}
        gene_set_to_beta[factor] = factor_to_beta[factor_int1]
        gene_set_factors.append(factor)
        current_values = []
        for factor_int2 in factor_to_label:
            factor2 = factor_int2
            factor2_label = factor_to_label[factor_int2]
            if factor == factor2:
                value = 1
            else:
                value = 0
            gene_set_factor_to_weight[factor][factor2] = value
            current_values.append(value)
        gene_sets.append(factor)
        gene_set_factor_matrix.append(current_values)
    gene_set_factor_matrix = np.array(gene_set_factor_matrix)

    gene_set_to_genes = {}
    max_gene_set_to_gene = -np.inf
    for gene in gene_factor_to_weight:
        for factor_int in gene_factor_to_weight[gene]:
            factor = factor_int
            factor_label = factor_to_label[factor_int]
            value = gene_factor_to_weight[gene][factor_int]
            if value < options.gene_min_loading:
                continue
            if factor not in gene_set_to_genes:
                gene_set_to_genes[factor] = {}

            gene_set_to_genes[factor][gene] = value
            max_gene_set_to_gene = np.maximum(max_gene_set_to_gene, value)

    gene_edge_full_width =  options.gene_edge_full_width if  options.gene_edge_full_width is not None else max_gene_set_to_gene

    for factor in gene_set_to_genes:
        for gene in gene_set_to_genes[factor]:
            value = gene_set_to_genes[factor][gene]
            if value > gene_edge_full_width:
                value = gene_edge_full_width                
            value /= gene_edge_full_width
            gene_set_to_genes[factor][gene] = value ** options.node_gene_loading_to_weight_pow


    gene_set_to_phenos = {}
    max_gene_set_to_pheno = -np.inf
    if add_phenos:
        for pheno in pheno_factor_to_weight:
            for factor_int in pheno_factor_to_weight[pheno]:
                factor = factor_int
                factor_label = factor_to_label[factor_int]
                value = pheno_factor_to_weight[pheno][factor_int]
                if value < options.pheno_min_loading:
                    continue

                if factor not in gene_set_to_phenos:
                    gene_set_to_phenos[factor] = {}

                gene_set_to_phenos[factor][pheno] = value
                max_gene_set_to_pheno = np.maximum(max_gene_set_to_pheno, value)

    pheno_edge_full_width =  options.pheno_edge_full_width if  options.pheno_edge_full_width is not None else max_gene_set_to_pheno

    for factor in gene_set_to_phenos:
        for pheno in gene_set_to_phenos[factor]:
            value = gene_set_to_phenos[factor][pheno]
            if value > pheno_edge_full_width:
                value = pheno_edge_full_width                
            value /= pheno_edge_full_width
            gene_set_to_phenos[factor][pheno] = value ** options.node_pheno_loading_to_weight_pow


    if add_genes:
        assert(len(gene_factors) == len(gene_set_factors))
    if add_phenos:
        assert(len(pheno_factors) == len(gene_set_factors))

    factor_matrix = gene_factor_matrix
    if add_phenos:
        if factor_matrix is None or np.prod(factor_matrix.shape) == 0:
            factor_matrix = pheno_factor_matrix
        elif np.prod(pheno_factor_matrix.shape) != 0:
            factor_matrix = np.vstack((factor_matrix, pheno_factor_matrix))

    if len(factor_matrix) == 0:
        bail("No gene factors survived the filters; exiting")

    #get the colors

    #edit here once we add in color by genes
    genes_mask = np.array([False for x in genes])

    if anchors is not None:
        colors = generate_distinct_colors(len(anchors), start_with_red_blue=options.colors_red_blue)
        color_labels = anchors
    else:
        colors = generate_distinct_colors(len(factors), start_with_red_blue=options.colors_red_blue)

        color_correlations = np.corrcoef(factor_matrix, rowvar=False)

        if len(colors) > 1:

            np.fill_diagonal(color_correlations, 0)

            #now rearrange the colors
            indices = [0]
            color_correlations[:,0] = -np.inf
            for i in range(1,len(colors)):
                next_factor = np.argmax(color_correlations[indices[i-1],:])
                color_correlations[:,next_factor] = -np.inf
                indices.append(next_factor)

            new_colors = [None] * len(colors)
            for i in range(len(indices)):
                new_colors[indices[i]] = colors[i]
            colors = new_colors

            factors_for_color = gene_factors if add_genes else pheno_factors
            color_labels = [factor_to_label[factors_for_color[i]] if factors_for_color[i] in factor_to_label else factors_for_color[i] for i in range(len(factors_for_color))]

    if options.pdf_out is not None or options.html_out is not None or options.json_out is not None:
        G = nx.Graph()
        add_nodes_and_edges(G, genes, gene_factor_to_weight, gene_to_combined_direct, gene_factors, phenos, pheno_factor_to_weight, pheno_to_combined_direct, pheno_factors, gene_sets, gene_set_to_label, gene_set_factor_to_weight, gene_set_to_beta, gene_set_factors, gene_set_to_genes, gene_set_to_phenos, colors, gene_pheno_to_weight, node_size_scale=options.node_size_scale, gene_node_size_further_scale=options.gene_node_further_scale, pheno_node_size_further_scale=options.pheno_node_further_scale, beta_node_size_further_scale=options.mech_node_further_scale, edge_max_width=options.edge_max_width, gene_node_opacity=options.gene_node_opacity, pheno_node_opacity=options.pheno_node_opacity, edge_opacity=options.edge_opacity, factor_to_anchor_relevance=factor_to_anchor_relevance, pheno_factor_to_anchor_weights=pheno_factor_to_anchor_weights, gene_factor_to_anchor_weights=gene_factor_to_anchor_weights, pheno_factor_to_anchor_relevance=pheno_factor_to_anchor_relevance, gene_factor_to_anchor_relevance=gene_factor_to_anchor_relevance)

        #calculate distance matrix
        full_factor_matrix = np.vstack((factor_matrix, gene_set_factor_matrix))

        weights = [gene_to_combined_direct[x][0] for x in genes] + [pheno_to_combined_direct[x][0] for x in phenos] + [gene_set_to_beta[x] for x in gene_sets]

        coordinates = weighted_mds(full_factor_matrix, weights=weights)

        coordinates[:,0] /= np.max(coordinates[:,0])
        coordinates[:,1] /= np.max(coordinates[:,1])

        coordinates *= options.coordinate_scale
        pos_bipartite = {}
        node_list = list(G.nodes)
        for i in range(len(node_list)):
            pos_bipartite[node_list[i]] = np.array([coordinates[i][0], coordinates[i][1]])

        # Save bipartite graph as PDF and HTML
        if options.pdf_out is not None:
            save_pdf(G, pos_bipartite, options.pdf_out, colors, color_labels, node_label_size_threshold=options.node_label_size_threshold, coordinate_scale=options.coordinate_scale, pdf_width=options.pdf_width, pdf_height=options.pdf_height, node_border_width=options.node_border_width, font_size=options.node_label_font_size)
        if options.html_out is not None:
            save_html(G, pos_bipartite, options.html_out, colors, color_labels, html_height=options.html_height, node_border_width=options.node_border_width, html_node_border_scale=options.html_node_border_scale, html_edge_width_scale=options.html_edge_width_scale, html_pos_scale=options.html_pos_scale, html_node_size_scale=options.html_node_size_scale, font_size=options.node_label_font_size, use_physics=options.use_physics, add_links=options.add_links)

        if options.json_out is not None:
            save_json(G, pos_bipartite, options.json_out, html_height=options.html_height, node_border_width=options.node_border_width, html_node_border_scale=options.html_node_border_scale, html_edge_width_scale=options.html_edge_width_scale, html_pos_scale=options.html_pos_scale, html_node_size_scale=options.html_node_size_scale, font_size=options.node_label_font_size)


if __name__ == '__main__':
    #cProfile.run('main()')
    main()


