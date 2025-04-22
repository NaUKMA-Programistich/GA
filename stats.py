import os
from typing import List, Dict, Any
import constants as C
import functions as F
import csv
import numpy as np

def get_detailed_header(count_runs: int) -> List[str]:
    header = [
        "Function",
        "Dimension",
        "Population",
        "Selection/Reproduction scheme",
        "Sampling/Deletion scheme",
        "Selection param/GG",
        "Crossover",
        "Pc",
        "Mutation",
        "Pm",
        "Encoding"
    ]

    for i in range(1, count_runs + 1):
        header.append(f"Run_{i}_IsSuc")
        header.append(f"Run_{i}_NI")
        header.append(f"Run_{i}_NFE")
        header.append(f"Run_{i}_F_max")
        header.append(f"Run_{i}_x_max")
        header.append(f"Run_{i}_F_avg")
        header.append(f"Run_{i}_FC")
        header.append(f"Run_{i}_PA")
        header.append(f"Run_{i}_DA")

    header.append(f"Stat")
    header.append(f"Suc")
    header.append(f"N_Suc")

    header.append(f"Min_NI")
    header.append(f"Max_NI")
    header.append(f"Avg_NI")
    header.append(f"Sigma_NI")

    header.append(f"Min_NFE")
    header.append(f"Max_NFE")
    header.append(f"Avg_NFE")
    header.append(f"Sigma_NFE")

    header.append(f"Min_F_max")
    header.append(f"Max_F_max")
    header.append(f"Avg_F_max")
    header.append(f"Sigma_F_max")

    header.append(f"Min_F_avg")
    header.append(f"Max_F_avg")
    header.append(f"Avg_F_avg")
    header.append(f"Sigma_F_avg")

    header.append(f"Min_FC")
    header.append(f"Max_FC")
    header.append(f"Avg_FC")
    header.append(f"Sigma_FC")

    header.append(f"Min_PA")
    header.append(f"Max_PA")
    header.append(f"Avg_PA")
    header.append(f"Sigma_PA")

    header.append(f"Min_DA")
    header.append(f"Max_DA")
    header.append(f"Avg_DA")
    header.append(f"Sigma_DA")

    header.append(f"Min_NI_f")
    header.append(f"Max_NI_f")
    header.append(f"Avg_NI_f")
    header.append(f"Sigma_NI_f")

    header.append(f"Min_NFE_f")
    header.append(f"Max_NFE_f")
    header.append(f"Avg_NFE_f")
    header.append(f"Sigma_NFE_f")

    header.append(f"Min_F_max_f")
    header.append(f"Max_F_max_f")
    header.append(f"Avg_F_max_f")
    header.append(f"Sigma_F_max_f")

    header.append(f"Min_F_avg_f")
    header.append(f"Max_F_avg_f")
    header.append(f"Avg_F_avg_f")
    header.append(f"Sigma_F_avg_f")

    header.append(f"Min_FC_f")
    header.append(f"Max_FC_f")
    header.append(f"Avg_FC_f")
    header.append(f"Sigma_FC_f")

    header.append(f"Min_PA_f")
    header.append(f"Max_PA_f")
    header.append(f"Avg_PA_f")
    header.append(f"Sigma_PA_f")

    header.append(f"Min_DA_f")
    header.append(f"Max_DA_f")
    header.append(f"Avg_DA_f")
    header.append(f"Sigma_DA_f")

    return header

def get_summary_header() -> List[str]:
    header = [
        "Function",
        "Dimension",
        "Population",
        "Selection/Reproduction scheme",
        "Sampling/Deletion scheme",
        "Selection param/GG",
        "Crossover",
        "Pc",
        "Mutation",
        "Pm",
        "Encoding"
    ]

    header.append(f"Success rate")

    header.append(f"Min_NI")
    header.append(f"Max_NI")
    header.append(f"Avg_NI")
    header.append(f"Sigma_NI")

    header.append(f"Min_NFE")
    header.append(f"Max_NFE")
    header.append(f"Avg_NFE")
    header.append(f"Sigma_NFE")

    header.append(f"Min_F_max")
    header.append(f"Max_F_max")
    header.append(f"Avg_F_max")
    header.append(f"Sigma_F_max")

    header.append(f"Min_F_avg")
    header.append(f"Max_F_avg")
    header.append(f"Avg_F_avg")
    header.append(f"Sigma_F_avg")

    header.append(f"Min_FC")
    header.append(f"Max_FC")
    header.append(f"Avg_FC")
    header.append(f"Sigma_FC")

    header.append(f"Min_PA")
    header.append(f"Max_PA")
    header.append(f"Avg_PA")
    header.append(f"Sigma_PA")

    header.append(f"Min_DA")
    header.append(f"Max_DA")
    header.append(f"Avg_DA")
    header.append(f"Sigma_DA")

    header.append(f"Min_NI_f")
    header.append(f"Max_NI_f")
    header.append(f"Avg_NI_f")
    header.append(f"Sigma_NI_f")

    header.append(f"Min_NFE_f")
    header.append(f"Max_NFE_f")
    header.append(f"Avg_NFE_f")
    header.append(f"Sigma_NFE_f")

    header.append(f"Min_F_max_f")
    header.append(f"Max_F_max_f")
    header.append(f"Avg_F_max_f")
    header.append(f"Sigma_F_max_f")

    header.append(f"Min_F_avg_f")
    header.append(f"Max_F_avg_f")
    header.append(f"Avg_F_avg_f")
    header.append(f"Sigma_F_avg_f")

    header.append(f"Min_FC_f")
    header.append(f"Max_FC_f")
    header.append(f"Avg_FC_f")
    header.append(f"Sigma_FC_f")

    header.append(f"Min_PA_f")
    header.append(f"Max_PA_f")
    header.append(f"Avg_PA_f")
    header.append(f"Sigma_PA_f")

    header.append(f"Min_DA_f")
    header.append(f"Max_DA_f")
    header.append(f"Avg_DA_f")
    header.append(f"Sigma_DA_f")

    return header

def prepare_detail_file(repro_type, dim, pop_size, count_runs):
    detailed_stats_file = C.DETAILED_STATS_FILENAME_TEMPLATE.format(
        repro_type=repro_type, dim=dim, pop_size=pop_size)
    if os.path.exists(detailed_stats_file):
        os.remove(detailed_stats_file)

    header = get_detailed_header(count_runs)
    with open(detailed_stats_file, 'w') as file:
        file.write(','.join(header) + '\n')

def prepare_summary_file(repro_type, dim, pop_size):
    summary_stats_file = C.SUMMARY_STATS_FILENAME_TEMPLATE.format(
        repro_type=repro_type, dim=dim, pop_size=pop_size)
    if os.path.exists(summary_stats_file):
        os.remove(summary_stats_file)

    header = get_summary_header()
    with open(summary_stats_file, 'w') as file:
        file.write(','.join(header) + '\n')

def write_to_files(
    repro_type: 
    str, 
    dim: int, 
    pop_size: int, 
    count_runs: int,
    encoding_type: str,
    crossover_type: str,
    mutation_type: str,
    selection_method: str,
    cx_pb: float,
    mut_pb: float,
    results: List[Dict[str, Any]]
) -> None:
    print(f"Writing to files for {repro_type} with dim={dim}, pop_size={pop_size}, count_runs={count_runs}")

    detailed_stats_file = C.DETAILED_STATS_FILENAME_TEMPLATE.format(repro_type=repro_type, dim=dim, pop_size=pop_size)
    
    summary_stats_file = C.SUMMARY_STATS_FILENAME_TEMPLATE.format(repro_type=repro_type, dim=dim, pop_size=pop_size)
        
    selection_reproduction_scheme, sampling_deletion_scheme, selection_param_gg = F.parse_selection_scheme(selection_method)
    
    common_stats = [
        "Rastrigin",
        str(dim),
        str(pop_size),
        selection_reproduction_scheme,
        sampling_deletion_scheme,
        selection_param_gg,
        crossover_type,
        str(cx_pb),
        mutation_type,
        str(mut_pb),
        str(encoding_type),
    ]

    detailed_stats = []
    for result in results:
        is_success = 1 if result['success'] else 0
        ni = result['iterations']
        nfe = result['nfe']
        
        if result['best_fitness'] is not None:
            f_max = f"{result['best_fitness']:.6f}"
        else:
            f_max = "None"
            
        if isinstance(result['best_phenotype'], list):
            x_max = '[' + ', '.join(f"{x:.4f}" for x in result['best_phenotype']) + ']'
        else:
            x_max = str(result['best_phenotype'])
        
        if result['avg_fitness'] is not None:
            f_avg = f"{result['avg_fitness']:.6f}"
        else:
            f_avg = "None"
        
        fc = nfe
        pa = f"{result['peak_accuracy']:.6f}"
        da = f"{result['distance_accuracy']:.6f}"

        detailed_stats.append(str(is_success))
        detailed_stats.append(str(ni))
        detailed_stats.append(str(nfe))
        detailed_stats.append(str(f_max))
        detailed_stats.append(str(x_max))
        detailed_stats.append(str(f_avg))
        detailed_stats.append(str(fc))
        detailed_stats.append(str(pa))
        detailed_stats.append(str(da))

    success_results = [result for result in results if result["success"]]
    failed_results = [result for result in results if not result["success"]]

    summary_stats = [
        "Stat",
        str(len(success_results)),

        # NI
        "-" if not success_results else min(result['iterations'] for result in success_results),
        "-" if not success_results else max(result['iterations'] for result in success_results),
        "-" if not success_results else sum(result['iterations'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['iterations'] for result in success_results]),

        # NFE
        "-" if not success_results else min(result['nfe'] for result in success_results),
        "-" if not success_results else max(result['nfe'] for result in success_results),
        "-" if not success_results else sum(result['nfe'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['nfe'] for result in success_results]),

        # F_max
        "-" if not success_results else min(result['best_fitness'] for result in success_results),
        "-" if not success_results else max(result['best_fitness'] for result in success_results),
        "-" if not success_results else sum(result['best_fitness'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['best_fitness'] for result in success_results]),

        # F_avg
        "-" if not success_results else min(result['avg_fitness'] for result in success_results),
        "-" if not success_results else max(result['avg_fitness'] for result in success_results),
        "-" if not success_results else sum(result['avg_fitness'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['avg_fitness'] for result in success_results]),

        # FC
        "-" if not success_results else min(result['fc'] for result in success_results),
        "-" if not success_results else max(result['fc'] for result in success_results),
        "-" if not success_results else sum(result['fc'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['fc'] for result in success_results]),

        # PA
        "-" if not success_results else min(result['peak_accuracy'] for result in success_results),
        "-" if not success_results else max(result['peak_accuracy'] for result in success_results),
        "-" if not success_results else sum(result['peak_accuracy'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['peak_accuracy'] for result in success_results]),

        # DA
        "-" if not success_results else min(result['distance_accuracy'] for result in success_results),
        "-" if not success_results else max(result['distance_accuracy'] for result in success_results),
        "-" if not success_results else sum(result['distance_accuracy'] for result in success_results) / len(success_results),
        "-" if not success_results else np.std([result['distance_accuracy'] for result in success_results]),

        # NI_f
        "-" if not failed_results else min(result['iterations'] for result in failed_results),
        "-" if not failed_results else max(result['iterations'] for result in failed_results),
        "-" if not failed_results else sum(result['iterations'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['iterations'] for result in failed_results]),

        # NFE_f
        "-" if not failed_results else min(result['nfe'] for result in failed_results),
        "-" if not failed_results else max(result['nfe'] for result in failed_results),
        "-" if not failed_results else sum(result['nfe'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['nfe'] for result in failed_results]),

        # F_max_f
        "-" if not failed_results else min(result['best_fitness'] for result in failed_results),
        "-" if not failed_results else max(result['best_fitness'] for result in failed_results),
        "-" if not failed_results else sum(result['best_fitness'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['best_fitness'] for result in failed_results]),

        # F_avg_f
        "-" if not failed_results else min(result['avg_fitness'] for result in failed_results),
        "-" if not failed_results else max(result['avg_fitness'] for result in failed_results),
        "-" if not failed_results else sum(result['avg_fitness'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['avg_fitness'] for result in failed_results]),

        # FC_f
        "-" if not failed_results else min(result['fc'] for result in failed_results),
        "-" if not failed_results else max(result['fc'] for result in failed_results),
        "-" if not failed_results else sum(result['fc'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['fc'] for result in failed_results]),

        # PA_f
        "-" if not failed_results else min(result['peak_accuracy'] for result in failed_results),
        "-" if not failed_results else max(result['peak_accuracy'] for result in failed_results),
        "-" if not failed_results else sum(result['peak_accuracy'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['peak_accuracy'] for result in failed_results]),

        # DA_f
        "-" if not failed_results else min(result['distance_accuracy'] for result in failed_results),
        "-" if not failed_results else max(result['distance_accuracy'] for result in failed_results),
        "-" if not failed_results else sum(result['distance_accuracy'] for result in failed_results) / len(failed_results),
        "-" if not failed_results else np.std([result['distance_accuracy'] for result in failed_results]),
    ]
    
    
    common_stats = [str(result) for result in common_stats]
    summary_stats = [str(result) for result in summary_stats]
    detailed_stats = [str(result) for result in detailed_stats]
    
    with open(detailed_stats_file, 'a') as file:
        file.write(','.join(common_stats + detailed_stats + summary_stats) + '\n')
        
    with open(summary_stats_file, 'a') as file:
        file.write(','.join(common_stats + summary_stats) + '\n')