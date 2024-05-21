def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table,evidence,evidence_value,target,target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)  #notice we are grabbing from t_subset, not entire table
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)  #count and divide
  return p_b_a

def cond_probs_product(table, evidence_row, target, target_value):
  #your function body below
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = []
  #cond_prob_list = [cond_prob(table, evidence_column, evidence_val, target, target_value) for evidence_column, evidence_val in evidence_complete]
  for pair in evidence_complete:
    evidence = pair[0]
    evidence_val = pair[1]
    cond_prob_list += [cond_prob(table, evidence, evidence_val, target, target_value)]
  
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the product of the list, finally multiply by P(Flu=0)
  neg_1 = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)

  #do same for P(Flu=1|...)
  pos_1 = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  neg,pos = compute_probs(neg_1, pos_1)
  #return your 2 results in a list
  return [neg, pos]
