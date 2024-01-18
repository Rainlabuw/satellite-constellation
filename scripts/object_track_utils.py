import numpy as np

def calc_pct_objects_tracked(assignments, task_objects, task_trans_state_dep_scaling_mat):
    """
    Calculates the percentage of objects tracked over time, given a sequence of assignments.

    TODO: add capability for init_assignment being provided (low priority)
    """
    T = len(assignments)
    n = assignments[0].shape[0]
    m = assignments[0].shape[1]

    num_tracking_opportunities = 0
    num_tracked = 0
    for task_object in task_objects:
        for k in range(T):
            if task_object.task_idxs[k] is not None:
                obj_task = task_object.task_idxs[k]
                num_tracking_opportunities += 1

                #If the object is tracked by any satellite in the task
                if np.max(assignments[k][:,obj_task]) == 1:
                    covering_sat = np.argmax(assignments[k][:,obj_task])

                    #Determine if the primary satellite missed the object due to handover
                    if k == 0: num_tracked += 1 #assume init_assignment=None, so primary satellite is always tracking
                    else:
                        prev_obj_task = assignments[k-1][covering_sat,:].nonzero()[0]

                        #if there is no handover between the previous task and the current one, then the primary satellite is tracking
                        if task_trans_state_dep_scaling_mat[prev_obj_task,obj_task] == 0:
                            num_tracked += 1
                else:
                    pass

    return num_tracked/num_tracking_opportunities

def object_tracking_history(assignments, task_objects, task_trans_state_dep_scaling_mat, sat_cover_matrix):
    T = len(assignments)
    n = assignments[0].shape[0]
    m = assignments[0].shape[1]

    rel_task_objects = [task_objects[45]]
    for i, task_object in enumerate(rel_task_objects):
        print(f"~~~~~~~Object {i}, going dir {task_object.dir}~~~~~~~")
        for k in range(1,T):
            if task_object.task_idxs[k] is not None:
                obj_task = task_object.task_idxs[k]
                obj_sec_task = obj_task + 49 #HARDCODED
                
                print(f"Timestep {k}, associated with tasks {obj_task} (and {obj_sec_task})")

                #associated sat
                if np.max(assignments[k][:,obj_task]) == 1:
                    prim_sat = np.argmax(assignments[k][:,obj_task])
                    print(f"\tPrim task: sat {prim_sat} task {obj_task} for {sat_cover_matrix[prim_sat, obj_task, k]} scaling")
                else:
                    prim_sat = None
                    print(f"\tPrim task not completed")
                if prim_sat is not None:
                    print(f"\tPrim sat: prev task {assignments[k-1][prim_sat,:].nonzero()[0]} for {sat_cover_matrix[prim_sat, obj_task, k-1]} scaling")

                #associated sec sat
                if np.max(assignments[k][:,obj_sec_task]) == 1:
                    sec_sat = np.argmax(assignments[k][:,obj_sec_task])
                    print(f"\tSec task: sat {sec_sat} task {obj_sec_task} for {sat_cover_matrix[sec_sat, obj_sec_task, k]} scaling")
                else:
                    sec_sat = None
                    print(f"\tSec task not completed")
                if sec_sat is not None:
                    print(f"\tSec sat: prev task {assignments[k-1][sec_sat,:].nonzero()[0]} for {sat_cover_matrix[sec_sat, obj_sec_task, k-1]} scaling")

if __name__ == "__main__":
    pass