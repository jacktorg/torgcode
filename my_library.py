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
  return p_b_a + .01

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

def metrics(zipped_list):
  #asserts here
  assert isinstance(zipped_list,list), f'Expecting a list within metrics, instead it is not a list.'
  for a,b in zipped_list:
   assert isinstance(a,(int,float)) and isinstance(b,(int,float)), f'zipped_list contains a non-int or non-float pair: {[a,b]}'
  for a,b in zipped_list:
   assert float(a) in [0.0,1.0] and float(b) in [0.0,1.0], f'zipped_list contains a non-binary pair: {[a,b]}'
  for pair in zipped_list:
    assert isinstance(pair,list), f'Expecting a list of lists within metrics, instead it is just a list.'
    assert len(pair) == 2, f'Expecting a list of pairs within metrics, instead it is not a list of pairs.'
  

  #body of function below
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  prec = 0 if (tp+fp) == 0 else tp/(tp+fp)
  recall = 0 if (tp+fn) == 0 else tp/(tp+fn)
  f1 = 0 if (prec+recall) == 0 else 2*((prec*recall)/(prec+recall))
  accuracy = sum([1 if pair[0]==pair[1] else 0 for pair in zipped_list])/len(zipped_list)

  #now build dictionary with the 4 measures
  metric_dict = {f'Precision': prec,
  'Recall': recall,
  'F1': f1,
  'Accuracy': accuracy}
  #finally, return the dictionary
  return metric_dict

def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use
  X = up_drop_column(train, target)
  y = up_get_column(train,target)  

  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)  

  clf = RandomForestClassifier(n, max_depth=2, random_state=0)

  clf.fit(X, y)  #builds the trees as specified above
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]  #probs is list of [neg,pos] like we are used to seeing.

  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)
  metrics_table


  return metrics_table


def try_archs(full_table, target, architectures, thresholds):
  #target is target column name
  #now loop through architectures
  for arch in architectures:
    all_results = up_neural_net(train, test, arch, target)

    #loop through thresholds
    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))

  return None  #main use is to print out threshold tables, not return anything useful.

#I'll give you a start

def try_archs(full_table, target, architectures, thresholds):
  #split full_table
  train, test = up_train_test_split(full_table, target, .4)

  #now loop through architectures
  for arch in architectures:
    all_results = up_neural_net(train, test, arch, target)

    #loop through thresholds
    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))

  return None  #main use is to print out threshold tables, not return anything useful.
