export default function ConfirmDialog({ open, title = "Confirm action", children, onConfirm, onCancel }) {
  if (!open) return null;
  return <div role="dialog" aria-modal="true" className="confirm-dialog">
    <h2>{title}</h2>
    <div>{children}</div>
    <div><button type="button" onClick={onCancel}>Cancel</button><button type="button" onClick={onConfirm}>Confirm</button></div>
  </div>;
}
