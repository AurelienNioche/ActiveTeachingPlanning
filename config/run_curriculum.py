import git

N_USERS = 5
N_ITEMS = 30
SEED_PARAM_LEARNERS = 123

REPO = git.Repo(search_parent_directories=True)
GIT_BRANCH = REPO.active_branch.name
GIT_HASH = REPO.head.commit.hexsha
COMMIT_NAME = GIT_BRANCH + '_' + GIT_HASH

EXPERIMENT_NAME = ''

BKP_FOLDER = "bkp/curriculum_runs"
FIG_FOLDER = "fig/curriculum_runs"

ALL_SESSION_LENGTHS = [(100, ), (20, 50), (20, 50, 100)]