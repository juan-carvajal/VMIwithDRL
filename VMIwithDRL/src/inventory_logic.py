from implementation.VMImodelVariable import VMI




if __name__ == '__main__':
    #unittest.main()
    initial_state = [10, 10, 10, 10, 10, 1, 0]
    # print(tensorflow.test.is_gpu_available())
    # magic numbers: runs 150 , eppercentage:0.25 , min_ep:0.05 , batch:32 , memory:1825

    # Estrategia buena para realizar el entrenmiento
    # agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=128, memory=1825, use_gpu=True,
    #                       epsilon_function='linear', min_epsilon=0.001, epsilon_min_percentage=0.25, lr=0.0005)

    train_runs = 1500
    model = VMI(4, 100, 5, train_runs, initial_state, 5, 100)
    current_state=initial_state
    from random import choice
    for i in range(365):
        hosp=[hosp.inventory.inventory for hosp in model.hospitals]
        action=choice(list(model.valid_actions(current_state)))
        A = action // 11
        prep_donors = int((((action % 11) * 10) / 100.0) * 100)
        state, action, next_state, reward, term=model.model_logic(current_state,action)
        current_state=next_state
        #print(state,hosp,(A,prep_donors))
        print(state[:-2],'-->',next_state[:-2],(A,prep_donors))



