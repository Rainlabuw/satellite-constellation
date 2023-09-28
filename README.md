# RAIN-Lab

Welcome to the official RAIN Lab GitHub repo. 

There are 2 main directories with self-explanatory names: `/active_projects` and `/inactive_projects`. Projects should be moved to their associated directory.

Within these are project/topic directories of various projects at the RAIN lab. Each project directory should itself have sub-directories: `/scripts` and `/literature`. I recommend adding a directory in `/scripts` for each paper being worked on (if there are more than 1 papers). Writing a paper is a sub-project of the larger overall project. For example, the project is consensus algorithms, and within that project we are working on 2 separate papers; both papers deserve their own directories to hold their respective scripts.

Here is what our repo should look like:

```
.
├── /active_project/
│   └── /project_1/
│       ├── /scripts/
│       │   ├── /paper_1/
│       │   │   ├── /script1.py
│       │   │   ├── /script2.h
│       │   │   └── /script2.cpp
│       │   └── /paper_2/
│       │       └── /script3.py
│       └── /literature/
│           ├── /paper1.pdf
│           └── /paper2.pdf
└── /inactive_projects/
    └── /project_2/
        ├── /scripts/
        │   ├── /script4.py
        │   └── /script5.m
        └── /literature/
            └── /paper3.pdf
```

Note how /project_2/scripts does not have any sub-directories for papers since the creator only worked on one paper.

Create a separate markdown for Math
