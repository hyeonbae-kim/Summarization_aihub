from others.utils import rouge_results_to_str, test_rouge, tile

def _report_rouge(self, gold_path, can_path):
    # self.logger.info("Calculating Rouge")
    print("Calculating Rouge")
    results_dict = test_rouge('../temp', can_path, gold_path)
    return results_dict


if __name__ == '__main__':
    step = 20000
    gold_path = '../results'+'.%d.gold' % step
    can_path = '../results.'+'%d.candidate' % step

    rouges = _report_rouge(gold_path, can_path)
    print('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
    print('test/rouge1-F', rouges['rouge_1_f_score'], step)
    print('test/rouge2-F', rouges['rouge_2_f_score'], step)
    print('test/rougeL-F', rouges['rouge_l_f_score'], step)
