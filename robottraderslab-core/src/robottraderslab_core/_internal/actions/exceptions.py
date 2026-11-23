from .members import ActionID


class ActionError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ActionAlreadyExistsError(ActionError):
    def __init__(self, action_id: ActionID):
        self.action_id = action_id
        message = f"Action with ID '{action_id}' already exists."
        super().__init__(message)


class ActionNotFoundError(ActionError):
    def __init__(self, action_id: ActionID):
        self.action_id = action_id
        message = f"Action with ID '{action_id}' not found."
        super().__init__(message)
