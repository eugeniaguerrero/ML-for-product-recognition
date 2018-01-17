# Group Project

## Please, before adding a change:
1. `git pull master`
2. `git merge master`
3. `git status`
4. git `add <files>` or `git add .`

### Creating a branch
`git checkout -b <name of branch>`

### Switching to a branch
`git checkout <name of branch>`

## Add changes to the branch
`git push origin <name of branch>`

#When Upgrading the model
1. Update image dimensions in common.py
2. Update data stream target size
3. Update number of nodes on final layer to number of categories
4. Update the .compile function to add metrics=['accuracy']
5. Check the NN using the debug function!!!
6. Change the self.name of the model + ensure you don't delete clean_up_logs