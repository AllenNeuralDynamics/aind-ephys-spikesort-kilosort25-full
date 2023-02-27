
from subprocess import check_output



def get_git_commit_or_tag(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        tag = None
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        try:
            latest_tag = check_output(['git', 'show-ref', '--tags'], cwd=git_folder).decode('utf8').strip().split("\n")[-1]
        except:
            latest_tag = None
        if latest_tag is not None:
            tag_commit, tag_ref = latest_tag.split()
            if tag_commit == commit:
                tag = tag_ref.split("/")[-1]
        if tag is None:
            if shorten:
                commit = commit[:12]
    except:
        tag = None
        commit = None
    if tag is not None:
        return tag
    else:
        return commit
