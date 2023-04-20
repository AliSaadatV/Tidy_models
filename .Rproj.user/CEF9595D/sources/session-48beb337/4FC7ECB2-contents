### Load data
library(tidymodels)
tidymodels_prefer()
data(concrete, package = "modeldata")
glimpse(concrete)
#> Rows: 1,030
#> Columns: 9
#> $ cement               <dbl> 540.0, 540.0, 332.5, 332.5, 198.6, 266.0, 380.0, 380.…
#> $ blast_furnace_slag   <dbl> 0.0, 0.0, 142.5, 142.5, 132.4, 114.0, 95.0, 95.0, 114…
#> $ fly_ash              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ water                <dbl> 162, 162, 228, 228, 192, 228, 228, 228, 228, 228, 192…
#> $ superplasticizer     <dbl> 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0…
#> $ coarse_aggregate     <dbl> 1040.0, 1055.0, 932.0, 932.0, 978.4, 932.0, 932.0, 93…
#> $ fine_aggregate       <dbl> 676.0, 676.0, 594.0, 594.0, 825.5, 670.0, 594.0, 594.…
#> $ age                  <int> 28, 28, 270, 365, 360, 90, 365, 28, 28, 28, 90, 28, 2…
#> $ compressive_strength <dbl> 79.99, 61.89, 40.27, 41.05, 44.30, 47.03, 43.70, 36.4…

### average repeated measurements
concrete <- 
  concrete %>% 
  group_by(across(-compressive_strength)) %>% 
  summarize(compressive_strength = mean(compressive_strength),
            .groups = "drop")
nrow(concrete)
#> [1] 992

### split the data
set.seed(1501)
concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test  <- testing(concrete_split)

set.seed(1502)
concrete_folds <- 
  vfold_cv(concrete_train, strata = compressive_strength, repeats = 5)

### define recipes:
normalized_rec <- 
  recipe(compressive_strength ~ ., data = concrete_train) %>% 
  step_normalize(all_predictors()) 

poly_recipe <- 
  normalized_rec %>% 
  step_poly(all_predictors()) %>% 
  step_interact(~ all_predictors():all_predictors())

### define models
library(rules)
library(baguette)

linear_reg_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

nnet_spec <- 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
  set_engine("nnet", MaxNWts = 2600) %>% 
  set_mode("regression")

mars_spec <- 
  mars(prod_degree = tune()) %>%  #<- use GCV to choose terms
  set_engine("earth") %>% 
  set_mode("regression")

svm_r_spec <- 
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

svm_p_spec <- 
  svm_poly(cost = tune(), degree = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

knn_spec <- 
  nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

cart_spec <- 
  decision_tree(cost_complexity = tune(), min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

bag_cart_spec <- 
  bag_tree() %>% 
  set_engine("rpart", times = 50L) %>% 
  set_mode("regression")

rf_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

xgb_spec <- 
  boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
             min_n = tune(), sample_size = tune(), trees = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

cubist_spec <- 
  cubist_rules(committees = tune(), neighbors = tune()) %>% 
  set_engine("Cubist") 

### define grid search 
nnet_param <- 
  nnet_spec %>% 
  extract_parameter_set_dials() %>% 
  update(hidden_units = hidden_units(c(1, 27)))

### create workflow set
## normalized workflows
normalized <- 
  workflow_set(
    preproc = list(normalized = normalized_rec), 
    models = list(SVM_radial = svm_r_spec, SVM_poly = svm_p_spec, 
                  KNN = knn_spec, neural_network = nnet_spec)
  )
normalized
#> # A workflow set/tibble: 4 × 4
#>   wflow_id                  info             option    result    
#>   <chr>                     <list>           <list>    <list>    
#> 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 normalized_neural_network <tibble [1 × 4]> <opts[0]> <list [0]>

normalized %>% extract_workflow(id = "normalized_KNN")
#> ══ Workflow ═════════════════════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: nearest_neighbor()
#> 
#> ── Preprocessor ─────────────────────────────────────────────────────────────────────
#> 1 Recipe Step
#> 
#> • step_normalize()
#> 
#> ── Model ────────────────────────────────────────────────────────────────────────────
#> K-Nearest Neighbor Model Specification (regression)
#> 
#> Main Arguments:
#>   neighbors = tune()
#>   weight_func = tune()
#>   dist_power = tune()
#> 
#> Computational engine: kknn

normalized <- 
  normalized %>% 
  option_add(param_info = nnet_param, id = "normalized_neural_network")
normalized
#> # A workflow set/tibble: 4 × 4
#>   wflow_id                  info             option    result    
#>   <chr>                     <list>           <list>    <list>    
#> 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 normalized_neural_network <tibble [1 × 4]> <opts[1]> <list [0]>

## workflows without preprocessing
model_vars <- 
  workflow_variables(outcomes = compressive_strength, 
                     predictors = everything())

no_pre_proc <- 
  workflow_set(
    preproc = list(simple = model_vars), 
    models = list(MARS = mars_spec, CART = cart_spec, CART_bagged = bag_cart_spec,
                  RF = rf_spec, boosting = xgb_spec, Cubist = cubist_spec)
  )
no_pre_proc
#> # A workflow set/tibble: 6 × 4
#>   wflow_id           info             option    result    
#>   <chr>              <list>           <list>    <list>    
#> 1 simple_MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 simple_CART        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 simple_CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 simple_RF          <tibble [1 × 4]> <opts[0]> <list [0]>
#> 5 simple_boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 6 simple_Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>

## workflows with polynomial preprocessing
with_features <- 
  workflow_set(
    preproc = list(full_quad = poly_recipe), 
    models = list(linear_reg = linear_reg_spec, KNN = knn_spec)
  )

### combine all workflow sets
all_workflows <- 
  bind_rows(no_pre_proc, normalized, with_features) %>% 
  # Make the workflow ID's a little more simple: 
  mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))
all_workflows
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result    
#>   <chr>       <list>           <list>    <list>    
#> 1 MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 CART        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 RF          <tibble [1 × 4]> <opts[0]> <list [0]>
#> 5 boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 6 Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>
#> # ℹ 6 more rows

### option 1: Tune and evaluate results on all the grid search
grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

grid_results <-
  all_workflows %>%
  workflow_map(
    seed = 1503,
    resamples = concrete_folds,
    grid = 25,
    control = grid_ctrl
  )

grid_results
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result   
#>   <chr>       <list>           <list>    <list>   
#> 1 MARS        <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 2 CART        <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[3]> <rsmp[+]>
#> 4 RF          <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 5 boosting    <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 6 Cubist      <tibble [1 × 4]> <opts[3]> <tune[+]>
#> # ℹ 6 more rows

grid_results %>% 
  rank_results() %>% 
  filter(.metric == "rmse") %>% 
  select(model, .config, rmse = mean, rank)
#> # A tibble: 252 × 4
#>   model      .config                rmse  rank
#>   <chr>      <chr>                 <dbl> <int>
#> 1 boost_tree Preprocessor1_Model04  4.25     1
#> 2 boost_tree Preprocessor1_Model06  4.29     2
#> 3 boost_tree Preprocessor1_Model13  4.31     3
#> 4 boost_tree Preprocessor1_Model14  4.39     4
#> 5 boost_tree Preprocessor1_Model16  4.46     5
#> 6 boost_tree Preprocessor1_Model03  4.47     6
#> # ℹ 246 more rows

autoplot(
  grid_results,
  rank_metric = "rmse",  # <- how to order models
  metric = "rmse",       # <- which metric to visualize
  select_best = TRUE     # <- one point per workflow
) +
  geom_text(aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
  lims(y = c(3.5, 9.5)) +
  theme(legend.position = "none")

autoplot(grid_results, id = "Cubist", metric = "rmse")

### option 2: Tune with race_anova (part of the grid search)
library(finetune)

race_ctrl <-
  control_race(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

race_results <-
  all_workflows %>%
  workflow_map(
    "tune_race_anova",
    seed = 1503,
    resamples = concrete_folds,
    grid = 25,
    control = race_ctrl
  )

race_results
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result   
#>   <chr>       <list>           <list>    <list>   
#> 1 MARS        <tibble [1 × 4]> <opts[3]> <race[+]>
#> 2 CART        <tibble [1 × 4]> <opts[3]> <race[+]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[3]> <rsmp[+]>
#> 4 RF          <tibble [1 × 4]> <opts[3]> <race[+]>
#> 5 boosting    <tibble [1 × 4]> <opts[3]> <race[+]>
#> 6 Cubist      <tibble [1 × 4]> <opts[3]> <race[+]>
#> # ℹ 6 more rows

autoplot(
  race_results,
  rank_metric = "rmse",  
  metric = "rmse",       
  select_best = TRUE    
) +
  geom_text(aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
  lims(y = c(3.0, 9.5)) +
  theme(legend.position = "none")

### optional: compare complete grid search with race_anova
matched_results <- 
  rank_results(race_results, select_best = TRUE) %>% 
  select(wflow_id, .metric, race = mean, config_race = .config) %>% 
  inner_join(
    rank_results(grid_results, select_best = TRUE) %>% 
      select(wflow_id, .metric, complete = mean, 
             config_complete = .config, model),
    by = c("wflow_id", ".metric"),
  ) %>%  
  filter(.metric == "rmse")

library(ggrepel)

matched_results %>% 
  ggplot(aes(x = complete, y = race)) + 
  geom_abline(lty = 3) + 
  geom_point() + 
  geom_text_repel(aes(label = model)) +
  coord_obs_pred() + 
  labs(x = "Complete Grid RMSE", y = "Racing RMSE") 

### Finalizing a model
best_results <- 
  race_results %>% 
  extract_workflow_set_result("boosting") %>% 
  select_best(metric = "rmse")
best_results
#> # A tibble: 1 × 7
#>   trees min_n tree_depth learn_rate loss_reduction sample_size .config              
#>   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>                
#> 1  1957     8          7     0.0756    0.000000145       0.679 Preprocessor1_Model04

boosting_test_results <- 
  race_results %>% 
  extract_workflow("boosting") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(split = concrete_split)

collect_metrics(boosting_test_results)
#> # A tibble: 2 × 4
#>   .metric .estimator .estimate .config             
#>   <chr>   <chr>          <dbl> <chr>               
#> 1 rmse    standard       3.41  Preprocessor1_Model1
#> 2 rsq     standard       0.954 Preprocessor1_Model1

boosting_test_results %>% 
  collect_predictions() %>% 
  ggplot(aes(x = compressive_strength, y = .pred)) + 
  geom_abline(color = "gray50", lty = 2) + 
  geom_point(alpha = 0.5) + 
  coord_obs_pred() + 
  labs(x = "observed", y = "predicted")

### Optional: before finalizing the model, it's possible to see what model is working the best,
### then use iterative search (instead of grid search) to tune the parameters of the best model.
### more here: https://www.tmwr.org/iterative-search.html