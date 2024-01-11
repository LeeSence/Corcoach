function submit() {
  localStorage.setItem("value", 1);
}

function load_name() {
  localStorage.getItem("value");
  return (value = localStorage.getItem("value"));
}
