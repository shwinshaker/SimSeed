#!./env python
import os
import sys
import shutil
import re
import errno
import torch
import numpy as np
import bottleneck as bn
import warnings

__all__ = ['load_log', 'check_path', 'check_path_remote', 'save_model', 'save_checkpoint', 'save_array', 'get_files_regex_match']

def get_files_regex_match(path, regex_str='.*'):
    regex = re.compile(regex_str)
    files_match = []
    for _, _, files in os.walk(path):
        for file in files:
            if regex.match(file):
                files_match.append(file)
    assert(files_match), f'{regex} not found in {path}'
    return files_match

def save_array(path, arr, config=None):
    np.save(os.path.join(config.save_dir, path), arr)

def save_model(model, basename='model', config=None):
    torch.save(model.state_dict(), os.path.join(config.save_dir, '%s.pt' % basename))

def save_checkpoint(epoch, net, optimizer, scheduler=None, filename='checkpoint.pth.tar', config=None):
    filepath = os.path.join(config.save_dir, filename)
    state = {'epoch': epoch + 1,
             'state_dict': net.state_dict(),
             'optimizer' : optimizer.state_dict(),
            }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, filepath)


def load_log(logfile, nlogs=None, window=1, interval=None, min_count=1):
    if not os.path.isfile(logfile):
        warnings.warn('%s not found.' % logfile)
        return None

    with open(logfile, 'r') as f:
        header = f.readline().strip().split()
    data = np.loadtxt(logfile, skiprows=1)
    if data.size == 0:
        return None
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    if interval is not None:
        data = data[::interval]
    if not nlogs: nlogs = data.shape[0]
    data = data[:nlogs]

    def smooth(record):
        return bn.move_mean(record, window=window, min_count=min_count)
    return dict([(h, smooth(data[:, i])) for i, h in enumerate(header)])


def copyanything(path, src, dst):
    path_ = os.path.join(path, src)
    if os.path.isdir(path_):
        shutil.copytree(path_, os.path.join(dst, src))
    else:
        shutil.copy(path_, dst)

    # except OSError as exc: # python >2.5
    #     if exc.errno == errno.ENOTDIR:
    #         shutil.copy(src, dst)
    #     else: raise

def get_not_ext(exts, path='.'):
    files = []
    for f in os.listdir(path):
        if not any([f.endswith(ext) for ext in exts]):
            files.append(f)
    return files

def check_path(path, config):
    if not os.path.exists(path):
        return path

    print('> Path %s already exists.' % path)
    if os.path.exists('%s/train.out' % path):
        print('> Last 10 lines in train.out:')
        with open('%s/train.out' % path, 'r') as f:
            print(''.join(f.readlines()[-10:]))
    else:
        print('> train.out doesnt exist!')
    option = input('> Delete[d], Rename[r], Abort[a], Continue[c], Terminate[*]? ')
    if option.lower() == 'd':
        shutil.rmtree(path)
        return path

    if option.lower() == 'r':
        sufs = re.findall(r'-(\d+)$', path)
        if not sufs:
            path = path + '-1'
        else:
            i = int(sufs[0]) + 1
            path = re.sub(r'-(\d+)$', '-%i' %i, path)
        return check_path(path, config)

    if option.lower() == 'a':
        sys.exit(1)

    if option.lower() == 'c':
        assert(config['resume']), 'resume is not set in config!'
        # continue / resume
        save_dir = check_path(os.path.join(path, 'old'), config)
        exts = ['.pt', '.tar', 'old', '.npy']
        for f in get_not_ext(exts, path=path):
            if f.startswith('old'):
                continue
            if not f.endswith('.txt'):
                shutil.move(os.path.join(path, f), save_dir)
            else:
                copyanything(path, f, save_dir)
        return path

    sys.exit(2)


def check_path_remote(path, config, server=None, remote_dir=''):
    from fabric import Connection
    from patchwork import files
    connection = Connection(server)
    with connection.cd(remote_dir):
        if not files.exists(connection, path):
            # connection.run(f'mkdir {path}')
            return path

        print('> Path %s already exists.' % path)
        if files.exists(connection, f'{path}/train.out'):
            print('> Last 10 lines in train.out:')
            connection.run(f'cat {path}/train.out | tail -n 10')
        else:
            print('> train.out doesnt exist!')
        option = input('> Delete[d], Rename[r], Abort[a], Continue[c], Terminate[*]? ')
        if option.lower() == 'd':
            connection.run(f'rm -r {path}')
            print('> Path %s deleted!' % path)
            # connection.run(f'mkdir {path}')
            return path

        if option.lower() == 'r':
            # - append a suffix to filename
            raise NotImplementedError()
        #     sufs = re.findall(r'-(\d+)$', path)
        #     if not sufs:
        #         path = path + '-1'
        #     else:
        #         i = int(sufs[0]) + 1
        #         path = re.sub(r'-(\d+)$', '-%i' %i, path)
        #     return check_path(path, config)

        if option.lower() == 'a':
            sys.exit(1)

        if option.lower() == 'c':
            raise NotImplementedError()
            # - resume an interrupted job
        #     assert(config['resume']), 'resume is not set in config!'
        #     # continue / resume
        #     save_dir = check_path_remote(os.path.join(path, 'old'), config, server=server, remote_dir=remote_dir)
        #     exts = ['.pt', '.tar', 'old', '.npy']
        #     for f in get_not_ext(exts, path=path):
        #         if f.startswith('old'):
        #             continue
        #         if not f.endswith('.txt'):
        #             shutil.move(os.path.join(path, f), save_dir)
        #         else:
        #             copyanything(path, f, save_dir)
        #     return path

        sys.exit(2)

